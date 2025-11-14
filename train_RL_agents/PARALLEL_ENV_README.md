# Multicore Parallel Environment (SubprocVecEnv)

This module implements a subprocess-based vectorized environment architecture for efficient multicore data collection in reinforcement learning training.

## Overview

The `SubprocVecEnv` class enables running multiple environment instances in parallel subprocesses, allowing you to leverage multiple CPU cores for faster data collection during training. This is particularly useful for:

- Speeding up data collection for a single agent
- Better utilizing multi-core CPUs
- Maintaining independent environment states across subprocesses

## Architecture

- **`parallel_env.py`**: Contains `SubprocVecEnv` class and worker functions
- **Modified `trainer.py`**: Automatically detects and handles vectorized environments
- **Modified `train_RL_agents.py`**: Supports `num_parallel_envs` parameter in config files

## Usage

### 1. Configuration File

To enable parallel environments, add the `num_parallel_envs` parameter to your training configuration JSON file:

```json
{
    "seed": [0],
    "total_timesteps": 1000000,
    "eval_freq": 10000,
    "num_parallel_envs": 4,  // <-- Add this parameter
    "save_dir": "./training_data",
    "training_schedule": {...},
    "eval_schedule": {...},
    "imitation_learning": false,
    "agent_type": "IQN"
}
```

### 2. Running Training

Use the same training command as before:

```bash
python train_RL_agents.py -C config/your_config.json -D cpu
```

The trainer will automatically:
- Detect the `num_parallel_envs` parameter
- Create a vectorized environment with the specified number of parallel instances
- Collect experiences from all environments simultaneously
- Maintain backward compatibility if `num_parallel_envs` is not specified (defaults to 1)

### 3. Example Configurations

**Single Environment (Original Behavior)**:
```json
{
    "seed": [0],
    "total_timesteps": 1000000,
    // No num_parallel_envs parameter
    ...
}
```

**Parallel Environments (2 cores)**:
```json
{
    "seed": [0],
    "total_timesteps": 1000000,
    "num_parallel_envs": 2,  // Run 2 environments in parallel
    ...
}
```

**Parallel Environments (4 cores)**:
```json
{
    "seed": [0],
    "total_timesteps": 1000000,
    "num_parallel_envs": 4,  // Run 4 environments in parallel
    ...
}
```

## How It Works

### Environment Creation

When `num_parallel_envs > 1`:
1. Multiple environment factory functions are created with different seeds
2. Each environment runs in its own subprocess
3. Communication happens via `multiprocessing.Pipe()`

### Training Loop

1. **Action Collection**: Actions are collected for all robots across all parallel environments
2. **Parallel Execution**: All environments step simultaneously in their respective subprocesses
3. **Experience Collection**: Experiences from all environments are added to the shared replay buffer
4. **Independent Episodes**: Each environment maintains its own episode state and resets independently
5. **Learning**: Agent learns from the combined experiences in the replay buffer

### Key Features

- **Automatic Detection**: Trainer automatically detects and handles vectorized environments
- **Backward Compatible**: Works with existing single-environment configurations
- **Independent States**: Each subprocess maintains independent environment state
- **Shared Learning**: All experiences contribute to the same replay buffer and agent training
- **Error Handling**: Proper error propagation from subprocesses to main process

## Performance Considerations

- **Recommended Number of Parallel Environments**: 2-8 depending on your CPU cores
- **Memory Usage**: Each parallel environment requires its own memory space
- **CPU Utilization**: Best performance when `num_parallel_envs` ≤ number of physical CPU cores
- **Data Collection Speed**: Approximately linear speedup with number of parallel environments

## Implementation Details

### SubprocVecEnv Class

The `SubprocVecEnv` class manages multiple environment instances:

```python
class SubprocVecEnv:
    def reset(self) -> List[np.ndarray]:
        """Reset all environments and return initial observations"""
        
    def step(self, actions_list: List, is_continuous_action: bool):
        """Step all environments with their respective actions"""
        
    def get_robots_info(self) -> List[Dict]:
        """Get robot state information from all environments"""
        
    def close(self):
        """Close all subprocess environments"""
```

### Worker Function

Each subprocess runs a worker function that:
- Receives commands via Pipe: `step`, `reset`, `get_robots_info`, `get_env_info`, `close`
- Executes the commands on the local environment instance
- Sends results back to the main process
- Handles auto-reset when episodes end

### Modified Trainer

The `Trainer` class now includes:
- `_learn_single_env()`: Original single-environment training loop
- `_learn_vec_env()`: New vectorized environment training loop
- Automatic detection of environment type in `__init__()`
- `learn()` method dispatches to appropriate training loop

## Testing

Test configurations are provided in `config/`:
- `iqn_parallel_test.json`: Quick test with 2 parallel environments
- `iqn_single_test.json`: Single environment for comparison

Run tests:
```bash
# Test parallel environment
python train_RL_agents.py -C config/iqn_parallel_test.json -D cpu

# Test single environment (backward compatibility)
python train_RL_agents.py -C config/iqn_single_test.json -D cpu
```

## Dependencies

- `cloudpickle`: For serializing environment creation functions
- `multiprocessing`: Python standard library for subprocess management
- `numba`: For fast dynamics computation (already required)

## Notes

- Each parallel environment uses a different seed: `base_seed + i * 1000` where `i` is the environment index
- Evaluation always uses a single environment (`eval_env`) for consistency
- Verbose logging is reduced for parallel environments (only first env logs episode info)
- The implementation maintains the same training dynamics as single-environment training
