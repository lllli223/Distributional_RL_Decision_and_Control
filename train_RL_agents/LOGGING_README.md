# Multiprocess-Safe Logging Infrastructure

This document describes the multiprocess-safe logging infrastructure implemented for the RL training pipeline.

## Overview

The logging system provides:
- **Queue-based log aggregation** for multiprocess safety
- **Centralized listener thread** for log writing
- **Rotating file handlers** with separate files for different log levels
- **Worker logger configuration** for subprocess environments
- **Integration with Trainer** for comprehensive training metrics logging

## Components

### 1. MultiprocessLogger (`multiprocess_logger.py`)

The core logging class that provides multiprocess-safe logging capabilities.

#### Features
- Aggregates logs from multiple processes using a shared queue
- Centralized listener thread handles all file I/O
- Rotating file handlers prevent logs from growing indefinitely
- Separate log files for:
  - `training_*.log`: All INFO+ level logs
  - `errors_*.log`: WARNING+ level logs
  - `debug_*.log`: DEBUG+ level logs with detailed information
  - Console output for WARNING+ messages

#### Basic Usage

```python
from multiprocess_logger import MultiprocessLogger

# Create logger
logger = MultiprocessLogger(
    log_dir="./logs/experiment_001",
    name="my_logger",
    level=logging.INFO,
    max_bytes=10*1024*1024,  # 10MB per file
    backup_count=5  # Keep 5 backup files
)

# Start logging
logger.start()

# Use logging methods
logger.log_episode_start(env_idx=0, episode_num=1, num_robots=3, timestep=0)
logger.log_episode_end(env_idx=0, episode_num=1, episode_length=100, 
                       rewards=[10.5, 15.2], success_info={0: True, 1: True},
                       timestep=100)

# Stop logging
logger.stop()
```

#### Context Manager Usage

```python
with MultiprocessLogger(log_dir="./logs/run_001") as logger:
    # Use logger
    logger.log_episode_start(...)
    # Automatically stopped on exit
```

### 2. Parallel Environment Integration (`parallel_env.py`)

The `SubprocVecEnv` class now supports logging from worker processes.

#### Changes
- `worker` function accepts optional `log_queue` parameter
- Worker processes configure logger on initialization
- `SubprocVecEnv.__init__` accepts optional `log_queue` parameter
- Log queue is passed to all worker processes
- Cleanup logic handles logging shutdown

#### Usage

```python
from parallel_env import SubprocVecEnv
from multiprocess_logger import MultiprocessLogger

# Create logger
logger = MultiprocessLogger(log_dir="./logs/parallel_run")
logger.start()

# Create vectorized environment with log queue
vec_env = SubprocVecEnv(
    env_fns=[lambda: create_env() for _ in range(4)],
    log_queue=logger.log_queue
)

# Use environment...
observations = vec_env.reset()
next_obs, rewards, dones, infos = vec_env.step(actions)

# Clean up
vec_env.close()
logger.stop()
```

### 3. Trainer Integration (`policy/trainer.py`)

The `Trainer` class now supports comprehensive logging of training metrics.

#### Changes
- Added optional `logger` parameter to `__init__`
- Automatic detection of vectorized vs single environment
- Episode logging:
  - Episode start events
  - Episode completion with rewards and success status
  - Robot deactivation events (collision/goal reached)
- Performance metrics logging:
  - Evaluation results
  - Model checkpoint saves
- Works with both single and vectorized environments

#### Usage

```python
from policy.trainer import Trainer
from multiprocess_logger import MultiprocessLogger

# Create logger
logger = MultiprocessLogger(log_dir="./logs/training_run")
logger.start()

# Create trainer with logger
trainer = Trainer(
    train_env=train_env,
    eval_env=eval_env,
    eval_schedule=eval_schedule,
    rl_agent=agent,
    logger=logger  # Pass logger to trainer
)

# Train with automatic logging
trainer.learn(
    total_timesteps=100000,
    eval_freq=5000,
    eval_log_path="./checkpoints"
)

logger.stop()
```

## Logged Events

### Episode Events
- **Episode Start**: Environment index, episode number, number of robots, timestep
- **Episode End**: Environment index, episode number, length, rewards, success status, timestep
- **Robot Deactivation**: Environment index, robot index, reason (collision/goal_reached), episode step, timestep

### Training Metrics
- **Training Step**: Timestep, loss (optional), exploration rate (optional)
- **Evaluation**: Timestep, average reward, success rate, average time, average energy
- **Checkpoint**: Timestep, checkpoint path
- **Timing**: Operation name, duration, timestep (optional)

## Log Format

### Standard Format (training.log, errors.log)
```
YYYY-MM-DD HH:MM:SS,mmm - ProcessName(PID) - LEVEL - message
```

Example:
```
2025-11-14 08:00:31,183 - MainProcess(7969) - INFO - Episode started - Env: 0, Episode: 1, Robots: 3, Timestep: 0
```

### Detailed Format (debug.log)
```
YYYY-MM-DD HH:MM:SS,mmm - name - ProcessName(PID) - ThreadName - LEVEL - filename:lineno - message
```

## Best Practices

### 1. Always Use Context Manager (Recommended)
```python
with MultiprocessLogger(log_dir="./logs") as logger:
    # Your code here
    pass
# Automatic cleanup
```

### 2. Pass Log Queue to Vectorized Environments
```python
vec_env = SubprocVecEnv(
    env_fns=env_fns,
    log_queue=logger.log_queue  # Important for worker logging
)
```

### 3. Create Unique Log Directories
```python
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs/experiment_{timestamp}"
logger = MultiprocessLogger(log_dir=log_dir)
```

### 4. Check Log Files After Training
```bash
# View training log
tail -f logs/training_20251114_080031.log

# View errors only
tail -f logs/errors_20251114_080031.log

# View detailed debug logs
tail -f logs/debug_20251114_080031.log
```

## Complete Example

```python
import datetime
from multiprocess_logger import MultiprocessLogger
from parallel_env import SubprocVecEnv
from policy.trainer import Trainer

# Create unique log directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs/run_{timestamp}"

# Use context manager for automatic cleanup
with MultiprocessLogger(log_dir=log_dir) as logger:
    # Create vectorized environment
    vec_env = SubprocVecEnv(
        env_fns=[lambda: create_env() for _ in range(4)],
        log_queue=logger.log_queue
    )
    
    # Create trainer
    trainer = Trainer(
        train_env=vec_env,
        eval_env=eval_env,
        eval_schedule=eval_schedule,
        rl_agent=agent,
        logger=logger
    )
    
    # Train with automatic logging
    trainer.learn(
        total_timesteps=100000,
        eval_freq=5000,
        eval_log_path=f"./checkpoints/{timestamp}"
    )
    
    # Close environment
    vec_env.close()

print(f"Training complete. Logs saved to: {log_dir}")
```

## Backward Compatibility

The logging system is **fully backward compatible**:
- If no logger is passed to `Trainer`, it works as before (no logging)
- If no log_queue is passed to `SubprocVecEnv`, it works as before (no worker logging)
- Existing code will continue to work without any changes

## Performance Considerations

1. **Queue-based logging** adds minimal overhead (microseconds per log entry)
2. **Rotating file handlers** prevent disk space issues
3. **Separate files** for different log levels enable efficient log analysis
4. **Worker logging** (if enabled) uses the same queue, avoiding file contention
5. **Listener thread** handles all I/O asynchronously

## Troubleshooting

### Issue: Logs not appearing
- Ensure `logger.start()` is called before training
- Check that logger is passed to Trainer
- Verify log directory permissions

### Issue: Worker logs missing
- Ensure `log_queue` is passed to `SubprocVecEnv`
- Check that logger is started before creating environments

### Issue: Log files too large
- Reduce `max_bytes` parameter
- Increase `backup_count` for more rotation
- Use more aggressive log level filtering

### Issue: Performance degradation
- Reduce logging frequency (avoid per-step logging in hot loops)
- Increase `max_bytes` to reduce rotation frequency
- Use appropriate log levels (DEBUG only when needed)

## API Reference

See `multiprocess_logger.py` for complete API documentation with all method signatures and parameters.
