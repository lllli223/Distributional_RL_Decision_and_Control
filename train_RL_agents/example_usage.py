"""
Example usage of the multiprocess-safe logging infrastructure.

This script demonstrates how to integrate the MultiprocessLogger with
the training pipeline for both single and vectorized environments.
"""

# Example 1: Single Environment with Logging
def example_single_environment():
    """
    Example of using MultiprocessLogger with a single environment.
    """
    from multiprocess_logger import MultiprocessLogger
    # from policy.trainer import Trainer
    # from marinenav_env.marine_nav_env3 import MarineNavEnv3
    
    print("=" * 70)
    print("Example 1: Single Environment with Logging")
    print("=" * 70)
    
    # Create logger with context manager for automatic cleanup
    with MultiprocessLogger(log_dir="./logs/single_env_run") as logger:
        print("Logger started and ready")
        
        # Create environment (pseudocode - actual implementation needs proper imports)
        # env = MarineNavEnv3(...)
        # eval_env = MarineNavEnv3(...)
        
        # Create trainer with logger
        # trainer = Trainer(
        #     train_env=env,
        #     eval_env=eval_env,
        #     eval_schedule=eval_schedule,
        #     rl_agent=agent,
        #     logger=logger  # Pass logger to trainer
        # )
        
        # Train with automatic logging
        # trainer.learn(
        #     total_timesteps=100000,
        #     eval_freq=5000,
        #     eval_log_path="./checkpoints/single_env"
        # )
        
        print("Training would happen here with automatic logging")
        print("Logs will be saved to: ./logs/single_env_run/")
        print("  - training_*.log: All info and above")
        print("  - errors_*.log: Warnings and errors only")
        print("  - debug_*.log: Detailed debug information")
    
    print("Logger automatically stopped on exit\n")


# Example 2: Vectorized Environment with Logging
def example_vectorized_environment():
    """
    Example of using MultiprocessLogger with vectorized environments.
    """
    from multiprocess_logger import MultiprocessLogger
    # from parallel_env import SubprocVecEnv
    # from policy.trainer import Trainer
    
    print("=" * 70)
    print("Example 2: Vectorized Environment with Logging")
    print("=" * 70)
    
    # Create logger
    with MultiprocessLogger(log_dir="./logs/parallel_env_run") as logger:
        print("Logger started and ready")
        
        # Create vectorized environment with log queue
        # vec_env = SubprocVecEnv(
        #     env_fns=[lambda: create_env() for _ in range(4)],
        #     log_queue=logger.log_queue  # Pass log queue for worker logging
        # )
        
        # Create trainer with logger
        # trainer = Trainer(
        #     train_env=vec_env,
        #     eval_env=eval_env,
        #     eval_schedule=eval_schedule,
        #     rl_agent=agent,
        #     logger=logger  # Pass logger to trainer
        # )
        
        # Train with automatic logging from all worker processes
        # trainer.learn(
        #     total_timesteps=100000,
        #     eval_freq=5000,
        #     eval_log_path="./checkpoints/parallel_env"
        # )
        
        print("Training would happen here with automatic logging")
        print("Logs from all 4 worker processes will be aggregated")
        print("Logs will be saved to: ./logs/parallel_env_run/")
        
        # Clean up
        # vec_env.close()
    
    print("Logger automatically stopped on exit\n")


# Example 3: Manual Logging Control
def example_manual_logging():
    """
    Example of manual logging control without context manager.
    """
    from multiprocess_logger import MultiprocessLogger
    
    print("=" * 70)
    print("Example 3: Manual Logging Control")
    print("=" * 70)
    
    # Create logger
    logger = MultiprocessLogger(
        log_dir="./logs/manual_run",
        name="manual_logger",
        max_bytes=5*1024*1024,  # 5MB per file
        backup_count=3  # Keep 3 backup files
    )
    
    # Start logger
    logger.start()
    print("Logger started manually")
    
    try:
        # Use logger for custom logging
        logger.log_episode_start(env_idx=0, episode_num=1, num_robots=3, timestep=0)
        logger.log_robot_deactivation(
            env_idx=0, robot_idx=0, reason="collision", 
            timestep=100, episode_step=50
        )
        logger.log_episode_end(
            env_idx=0, episode_num=1, episode_length=100,
            rewards=[10.5, 15.2, -5.3],
            success_info={0: False, 1: True, 2: False},
            timestep=200
        )
        logger.log_evaluation(
            timestep=1000,
            avg_reward=12.5,
            success_rate=0.75,
            avg_time=45.2,
            avg_energy=100.3
        )
        logger.log_checkpoint(
            timestep=1000,
            checkpoint_path="./checkpoints/model_1000.pth"
        )
        
        print("Custom logging completed")
        print("Logs saved to: ./logs/manual_run/")
        
    finally:
        # Always stop logger in finally block
        logger.stop()
        print("Logger stopped manually\n")


# Example 4: Configuration Best Practices
def example_best_practices():
    """
    Example showing best practices for logger configuration.
    """
    import datetime
    from multiprocess_logger import MultiprocessLogger
    
    print("=" * 70)
    print("Example 4: Configuration Best Practices")
    print("=" * 70)
    
    # Create unique log directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/run_{timestamp}"
    
    print(f"Creating logger with unique log directory: {log_dir}")
    
    # Configure logger with appropriate settings
    logger = MultiprocessLogger(
        log_dir=log_dir,
        name="training",
        level=20,  # logging.INFO
        max_bytes=10*1024*1024,  # 10MB per file
        backup_count=5  # Keep 5 backup files (50MB total per log type)
    )
    
    # Use context manager for automatic cleanup
    with logger:
        print("Logger configuration:")
        print(f"  - Log directory: {log_dir}")
        print(f"  - Max file size: 10MB")
        print(f"  - Backup count: 5")
        print(f"  - Total max size per log type: 50MB")
        print("\nTraining would happen here...")
    
    print(f"\nAfter training, check logs at: {log_dir}")
    print("Recommended: Use 'tail -f' to monitor logs during training")
    print(f"  tail -f {log_dir}/training_*.log\n")


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MultiprocessLogger Integration Examples")
    print("=" * 70 + "\n")
    
    example_single_environment()
    example_vectorized_environment()
    example_manual_logging()
    example_best_practices()
    
    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nKey Points:")
    print("1. Use context manager (with statement) for automatic cleanup")
    print("2. Pass logger to Trainer constructor")
    print("3. Pass log_queue to SubprocVecEnv for worker logging")
    print("4. Use unique log directories with timestamps")
    print("5. Configure appropriate file size and backup count")
    print("\nFor more details, see LOGGING_README.md")
