"""
Subprocess-based vectorized environment for multicore data collection.

This module provides a SubprocVecEnv class that runs multiple environment instances
in parallel subprocesses, allowing for efficient data collection across multiple CPU cores.
"""
import multiprocessing as mp
import cloudpickle


def worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker function that runs in a subprocess to handle environment interactions.
    
    Args:
        remote: The subprocess-side Pipe connection
        parent_remote: The parent-side Pipe connection (closed in subprocess)
        env_fn_wrapper: A callable that returns an environment instance
    """
    parent_remote.close()
    env = cloudpickle.loads(env_fn_wrapper)()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                try:
                    obs, rew, done, info = env.step(data["actions"], data["is_continuous"])
                    # Auto-reset on episode end
                    if all(done):
                        obs, _, _ = env.reset()
                    remote.send((obs, rew, done, info))
                except Exception as e:
                    import traceback
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    remote.send(("error", error_msg))
            elif cmd == "reset":
                obs, _, _ = env.reset()
                remote.send(obs)
            elif cmd == "get_robots_info":
                # Return robot information for data collection
                robots_info = []
                for rob in env.robots:
                    robots_info.append({
                        'deactivated': rob.deactivated,
                        'collision': rob.collision,
                        'reach_goal': rob.reach_goal
                    })
                remote.send(robots_info)
            elif cmd == "get_env_info":
                # Return environment state information
                info = {
                    'check_all_deactivated': env.check_all_deactivated(),
                    'num_robots': len(env.robots),
                    'episode_timesteps': env.episode_timesteps
                }
                remote.send(info)
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        remote.send(("error", error_msg))
        raise


class SubprocVecEnv:
    """
    Vectorized environment that runs multiple environment instances in parallel subprocesses.
    
    This allows for efficient data collection by running multiple environments concurrently
    across different CPU cores.
    
    Args:
        env_fns: List of functions that create environment instances
    """
    
    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        
        # Serialize environment creation functions
        env_fns_pickled = [cloudpickle.dumps(fn) for fn in env_fns]
        
        self.ps = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns_pickled):
            p = mp.Process(target=worker, args=(work_remote, remote, env_fn))
            p.daemon = True
            p.start()
            work_remote.close()
            self.ps.append(p)
        
        self.closed = False

    def reset(self):
        """Reset all environments and return initial observations."""
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions_list, is_continuous_action=False):
        """
        Step all environments with their respective actions.
        
        Args:
            actions_list: List of actions, one for each environment
            is_continuous_action: Whether actions are continuous
            
        Returns:
            Tuple of (observations, rewards, dones, infos) for all environments
        """
        for remote, act in zip(self.remotes, actions_list):
            remote.send(("step", {"actions": act, "is_continuous": is_continuous_action}))
        
        results = [remote.recv() for remote in self.remotes]
        
        # Check for errors
        for i, result in enumerate(results):
            if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
                raise RuntimeError(f"Error in subprocess {i}: {result[1]}")
        
        obs, rews, dones, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(infos)

    def get_robots_info(self):
        """Get robot information from all environments."""
        for remote in self.remotes:
            remote.send(("get_robots_info", None))
        return [remote.recv() for remote in self.remotes]

    def get_env_info(self):
        """Get environment state information from all environments."""
        for remote in self.remotes:
            remote.send(("get_env_info", None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        """Close all subprocess environments."""
        if self.closed:
            return
        
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except:
                pass
        
        for p in self.ps:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        self.closed = True

    def __del__(self):
        """Ensure environments are closed on deletion."""
        if not self.closed:
            self.close()
