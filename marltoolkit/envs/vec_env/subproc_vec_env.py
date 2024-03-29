import multiprocessing as mp
import sys
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import logger
from gymnasium.error import (AlreadyPendingCallError, ClosedEnvironmentError,
                             CustomSpaceError, NoAsyncCallError)
from gymnasium.vector.utils import (clear_mpi_env_vars, concatenate,
                                    create_empty_array, create_shared_memory,
                                    iterate, read_from_shared_memory,
                                    write_to_shared_memory)

from .base_vec_env import BaseVecEnv, CloudpickleWrapper


class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'
    WAITING_CALL = 'call'


class SubprocVecEnv(BaseVecEnv):
    """Creates a multiprocess vectorized wrapper for multiple environments,
    distributing each environment to its own process, allowing significant
    speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        obs_space: Optional[gym.Space] = None,
        state_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        start_method: Optional[str] = None,
        shared_memory: bool = True,
        copy: bool = True,
    ) -> None:
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.copy = copy
        self.shared_memory = shared_memory

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'

        ctx = mp.get_context(start_method)

        dummy_env = env_fns[0]()
        if (obs_space is None) or (action_space is None):
            obs_space = obs_space or dummy_env.obs_space
            state_space = state_space or dummy_env.state_space
            action_space = action_space or dummy_env.action_space

        dummy_env.close()
        del dummy_env

        super().__init__(len(env_fns), obs_space, state_space, action_space)
        print('*' * 100)
        print(self.single_action_space)

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(self.single_obs_space,
                                                   n=self.num_envs,
                                                   ctx=ctx)
                _state_buffer = create_shared_memory(self.single_state_space,
                                                     n=self.num_envs,
                                                     ctx=ctx)
                self.observations = read_from_shared_memory(
                    self.single_obs_space, _obs_buffer, n=self.num_envs)
                self.states = read_from_shared_memory(self.single_state_space,
                                                      _obs_buffer,
                                                      n=self.num_envs)
            except CustomSpaceError as e:
                raise ValueError(
                    'Using `shared_memory=True` in `AsyncVectorEnv` '
                    'is incompatible with non-standard Gymnasium observation spaces '
                    '(i.e. custom spaces inheriting from `gymnasium.Space`), and is '
                    'only compatible with default Gymnasium spaces (e.g. `Box`, '
                    '`Tuple`, `Dict`) for batching. Set `shared_memory=False` '
                    'if you use custom observation spaces.') from e
        else:
            _obs_buffer = None
            _state_buffer = None
            self.observations = create_empty_array(self.single_obs_space,
                                                   n=self.num_envs,
                                                   fn=np.zeros)
            self.states = create_empty_array(self.single_state_space,
                                             n=self.num_envs,
                                             fn=np.zeros)
        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _shareworker if self.shared_memory else _worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()

                args = (
                    idx,
                    CloudpickleWrapper(env_fn),
                    child_pipe,
                    parent_pipe,
                    _obs_buffer,
                    _state_buffer,
                    self.error_queue,
                )
                # daemon=True: if the main process crashes, we should not cause things to hang
                process = ctx.Process(
                    target=target,
                    name=f'Worker<{type(self).__name__}>-{idx}',
                    args=args,
                    daemon=True,
                )  # type: ignore[attr-defined]

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Send calls to the :obj:`reset` methods of the sub-environments.

        To get the results of these calls, you may invoke :meth:`reset_wait`.

        Args:
            seed: List of seeds for each environment
            options: The reset option

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: If the environment is already waiting for a pending call to another
                method (e.g. :meth:`step_async`). This can be caused by two consecutive
                calls to :meth:`reset_async`, with no call to :meth:`reset_wait` in between.
        """
        self._assert_is_running()

        if seed is None:
            self.seeds = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            self.seeds = [seed + i for i in range(self.num_envs)]
        assert len(self.seeds) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f'Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete',
                self._state.value,
            )
        # Seeds and options are only used once
        for pipe, single_seed in zip(self.parent_pipes, self.seeds):
            single_kwargs = {}
            if single_seed is not None:
                single_kwargs['seed'] = single_seed
            if options is not None:
                single_kwargs['options'] = options

            pipe.send(('reset', single_kwargs))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(
        self,
        timeout: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and
        returns the results.

        Args:
            timeout: Number of seconds before the call to `reset_wait` times out. If `None`, the call to `reset_wait` never times out.
            seed: ignored
            options: ignored

        Returns:
            A tuple of batched observations and list of dictionaries

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`reset_wait` was called without any prior call to :meth:`reset_async`.
            TimeoutError: If :meth:`reset_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                'Calling `reset_wait` without any prior '
                'call to `reset_async`.',
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f'The call to `reset_wait` has timed out after {timeout} second(s).'
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        infos = {}
        obs_lst, state_lst, info_data = zip(*results)
        for i, info in enumerate(info_data):
            infos = self._add_info(infos, info, i)

        if not self.shared_memory:
            self.observations = concatenate(self.single_obs_space, obs_lst,
                                            self.observations)
            self.states = concatenate(self.single_state_space, state_lst,
                                      self.states)
        return (
            (deepcopy(self.observations) if self.copy else self.observations),
            (deepcopy(self.states) if self.copy else self.states),
            infos,
        )

    def step_async(self, actions: np.ndarray) -> None:
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f'Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.',
                self._state.value,
            )
        actions = iterate(self.action_space, actions)
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout: Optional[Union[int, float]] = None):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                'Calling `step_wait` without any prior call '
                'to `step_async`.',
                AsyncState.WAITING_STEP.value,
            )
        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f'The call to `step_wait` has timed out after {timeout} second(s).'
            )

        obs_list, state_list, rewards, terminateds, truncateds, infos = (
            [],
            [],
            [],
            [],
            [],
            {},
        )
        successes = []
        for idx, pipe in enumerate(self.parent_pipes):
            result, success = zip(pipe.recv())
            successes.append(success)
            if success:
                obs, state, rew, terminated, truncated, info = result

                obs_list.append(obs)
                state_list.append(state)
                rewards.append(rew)
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos = self._add_info(infos, info, idx)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        if not self.shared_memory:
            self.observations = concatenate(
                self.single_obs_space,
                obs_list,
                self.observations,
            )
        return (
            deepcopy(self.observations) if self.copy else self.observations,
            deepcopy(self.states) if self.copy else self.states,
            np.array(rewards),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )

    def call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs
        to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                'Calling `call_async` while waiting '
                f'for a pending call to `{self._state.value}` to complete.',
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(('_call', (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout: Optional[Union[int, float]] = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `step_wait` times out.
                If `None` (default), the call to `step_wait` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_wait` without any prior call to `call_async`.
            TimeoutError: The call to `call_wait` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                'Calling `call_wait` without any prior call to `call_async`.',
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f'The call to `call_wait` has timed out after {timeout} second(s).'
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def close_extras(self,
                     timeout: Optional[Union[int, float]] = None,
                     terminate: bool = False):
        """Close the environments & clean up the extra resources (processes and
        pipes).

        Args:
            timeout: Number of seconds before the call to :meth:`close` times out. If ``None``,
                the call to :meth:`close` never times out. If the call to :meth:`close`
                times out, then all processes are terminated.
            terminate: If ``True``, then the :meth:`close` operation is forced and all processes are terminated.

        Raises:
            TimeoutError: If :meth:`close` timed out.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    f'Calling `close` while waiting for a pending call to `{self._state.value}` to complete.'
                )
                function = getattr(self, f'{self._state.value}_wait')
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(('close', None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def set_attr(
        self,
        name: str,
        values: Any,
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                'Values must be a list or tuple with length equal to the '
                f'number of environments. Got `{len(values)}` values for '
                f'{self.num_envs} environments.')

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                'Calling `set_attr` while waiting '
                f'for a pending call to `{self._state.value}` to complete.',
                self._state.value,
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(('_setattr', (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def _check_spaces(self):
        self._assert_is_running()
        spaces = (
            self.single_obs_space,
            self.single_state_space,
            self.single_action_space,
        )
        for pipe in self.parent_pipes:
            pipe.send(('_check_spaces', spaces))
        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        same_obs_spaces, same_state_spaces, same_action_spaces = zip(*results)
        if not all(same_obs_spaces):
            raise RuntimeError(
                'Some environments have an observation space different from '
                f'`{self.single_obs_space}`. In order to batch observations, '
                'the observation spaces from all environments must be equal.')
        if not all(same_state_spaces):
            raise RuntimeError(
                'Some environments have an observation space different from '
                f'`{self.single_state_space}`. In order to batch observations, '
                'the observation spaces from all environments must be equal.')
        if not all(same_action_spaces):
            raise RuntimeError(
                'Some environments have an action space different from '
                f'`{self.single_action_space}`. In order to batch actions, the '
                'action spaces from all environments must be equal.')

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                f'Trying to operate on `{type(self).__name__}`, after a call to `close()`.'
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(
                f'Received the following error from Worker-{index}: {exctype.__name__}: {value}'
            )
            logger.error(f'Shutting down Worker-{index}.')
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                logger.error(
                    'Raising the last exception back to the main process.')
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector environment is
        closed."""
        if not getattr(self, 'closed', True) and hasattr(self, '_state'):
            self.close(terminate=True)


def _worker(
    index: int,
    pipe: mp.connection.Connection,
    parent_pipe: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    shared_memory: bool,
    error_queue: mp.Queue,
) -> None:
    # Import here to avoid a circular import
    parent_pipe.close()
    env = env_fn_wrapper.x()
    assert shared_memory is None
    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            command, data = pipe.recv()

            if command == 'reset':
                state, obs, reset_info = env.reset(**data)
                pipe.send((state, obs, reset_info), True)

            elif command == 'step':
                (state, obs, reward, terminated, truncated,
                 info) = env.step(data)
                done = terminated or truncated
                if done:
                    old_obs, old_state, old_info = obs, state, info
                    state, obs, info = env.reset()
                    # save final obs where user can get it, then reset
                    info['final_obs'] = old_obs
                    info['final_state'] = old_state
                    info['final_info'] = old_info
                pipe.send((state, obs, reward, done, info, reset_info))

            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))

            elif command == 'close':
                pipe.send((None, True))
                env.close()
                pipe.close()
                break

            elif command == '_call':
                name, args, kwargs = data
                if name in ['reset', 'step', 'seed', 'close']:
                    raise ValueError(
                        f'Trying to call function `{name}` with '
                        f'`_call`. Use `{name}` directly instead.')
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))

            elif command == '_setattr':
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == '_check_spaces':
                pipe.send((
                    (
                        data[0] == env.obs_space,
                        data[1] == env.state_space,
                        data[2] == env.action_space,
                    ),
                    True,
                ))
            else:
                raise RuntimeError(
                    f'Received unknown command `{command}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, `_call`, '
                    '`_setattr`, `_check_spaces`}.')
        except (KeyboardInterrupt, Exception):
            error_queue.put((index, ) + sys.exc_info()[:2])
            pipe.send((None, False))
        finally:
            env.close()


def _shareworker(
    index: int,
    pipe: mp.connection.Connection,
    parent_pipe: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    shared_memory: bool,
    error_queue: mp.Queue,
) -> None:
    parent_pipe.close()
    env = env_fn_wrapper.x()
    obs_space = env.obs_space
    state_space = env.state_space

    while True:
        try:
            command, data = pipe.recv()

            if command == 'reset':
                obs, state, info = env.reset(**data)
                write_to_shared_memory(obs_space, index, obs, shared_memory)
                write_to_shared_memory(state_space, index, state,
                                       shared_memory)

                pipe.send(((obs, state, info), True))

            if command == 'step':
                obs, state, reward, terminated, truncated, info = env.step(
                    data)
                done = terminated or truncated
                if done:
                    old_obs, old_state, old_info = obs, state, info
                    obs, state, info = env.reset()
                    info['final_obs'] = old_obs
                    info['final_state'] = old_state
                    info['final_info'] = old_info
                write_to_shared_memory(obs_space, index, obs, shared_memory)
                write_to_shared_memory(state_space, index, state,
                                       shared_memory)

                pipe.send((obs, state, reward, done, info), True)

            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))

            elif command == 'close':
                pipe.send((None, True))
                env.close()
                pipe.close()
                break

            elif command == '_call':
                name, args, kwargs = data
                if name in ['reset', 'step', 'seed', 'close']:
                    raise ValueError(
                        f'Trying to call function `{name}` with '
                        f'`_call`. Use `{name}` directly instead.')
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))

            elif command == '_setattr':
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == '_check_spaces':
                pipe.send((
                    (
                        data[0] == obs_space,
                        data[1] == state_space,
                        data[2] == env.action_space,
                    ),
                    True,
                ))
            else:
                raise RuntimeError(
                    f'Received unknown command `{command}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, `_call`, '
                    '`_setattr`, `_check_spaces`}.')
        except (KeyboardInterrupt, Exception):
            error_queue.put((index, ) + sys.exc_info()[:2])
            pipe.send((None, False))
        finally:
            env.close()
