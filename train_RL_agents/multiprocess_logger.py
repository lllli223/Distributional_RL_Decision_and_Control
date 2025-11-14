"""
Multiprocess-safe logging infrastructure for parallel training environments.

This module provides a MultiprocessLogger class that aggregates logs from multiple
subprocesses using a queue-based approach, with a centralized listener thread for
log writing and rotating file handlers for different log levels.
"""
import logging
import logging.handlers
import multiprocessing as mp
import os
import threading
from datetime import datetime


class MultiprocessLogger:
    """
    Multiprocess-safe logger with queue-based log aggregation.
    
    This class creates a centralized logging system that can safely receive logs
    from multiple subprocesses. It uses a queue to aggregate logs and a listener
    thread to write them to files with rotation support.
    
    Args:
        log_dir: Directory where log files will be saved
        name: Logger name (default: "multiprocess_logger")
        level: Logging level (default: logging.INFO)
        max_bytes: Maximum size of each log file before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
    """
    
    def __init__(self, 
                 log_dir,
                 name="multiprocess_logger",
                 level=logging.INFO,
                 max_bytes=10*1024*1024,
                 backup_count=5):
        
        self.log_dir = log_dir
        self.name = name
        self.level = level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create multiprocess-safe queue
        self.log_queue = mp.Queue(-1)
        
        # Listener thread for centralized log writing
        self.listener = None
        self.listener_thread = None
        
        # Flag to indicate if logging is active
        self.active = False
        
    def start(self):
        """Start the logging listener thread."""
        if self.active:
            return
        
        # Create listener with rotating file handlers
        self.listener = logging.handlers.QueueListener(
            self.log_queue,
            *self._create_handlers(),
            respect_handler_level=True
        )
        
        self.listener.start()
        self.active = True
        
    def stop(self):
        """Stop the logging listener thread and flush remaining logs."""
        if not self.active:
            return
        
        if self.listener:
            self.listener.stop()
        
        self.active = False
        
    def _create_handlers(self):
        """Create rotating file handlers for different log levels."""
        handlers = []
        
        # Create timestamp for log file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Handler for all logs (INFO and above)
        all_log_path = os.path.join(self.log_dir, f"training_{timestamp}.log")
        all_handler = logging.handlers.RotatingFileHandler(
            all_log_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        all_handler.setLevel(logging.INFO)
        all_handler.setFormatter(self._create_formatter())
        handlers.append(all_handler)
        
        # Handler for warning and error logs
        error_log_path = os.path.join(self.log_dir, f"errors_{timestamp}.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(self._create_formatter())
        handlers.append(error_handler)
        
        # Handler for debug logs (separate file for detailed debugging)
        debug_log_path = os.path.join(self.log_dir, f"debug_{timestamp}.log")
        debug_handler = logging.handlers.RotatingFileHandler(
            debug_log_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(self._create_formatter(detailed=True))
        handlers.append(debug_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(self._create_formatter())
        handlers.append(console_handler)
        
        return handlers
    
    def _create_formatter(self, detailed=False):
        """Create log formatter."""
        if detailed:
            format_str = '%(asctime)s - %(name)s - %(processName)s(%(process)d) - %(threadName)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        else:
            format_str = '%(asctime)s - %(processName)s(%(process)d) - %(levelname)s - %(message)s'
        
        return logging.Formatter(format_str)
    
    def get_logger(self, name=None):
        """
        Get a logger instance for the main process.
        
        Args:
            name: Logger name (default: uses the MultiprocessLogger's name)
            
        Returns:
            Logger instance configured to use the queue handler
        """
        logger_name = name if name else self.name
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # Add queue handler
        queue_handler = logging.handlers.QueueHandler(self.log_queue)
        logger.addHandler(queue_handler)
        
        return logger
    
    def configure_worker_logger(self, name=None):
        """
        Configure logger for a worker subprocess.
        
        This should be called in each worker process to set up logging
        that sends logs to the queue for centralized handling.
        
        Args:
            name: Logger name (default: uses the MultiprocessLogger's name)
            
        Returns:
            Logger instance configured for the worker process
        """
        logger_name = name if name else self.name
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # Add queue handler that sends logs to the main process
        queue_handler = logging.handlers.QueueHandler(self.log_queue)
        logger.addHandler(queue_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def log_episode_start(self, env_idx, episode_num, num_robots, timestep):
        """
        Log episode start event.
        
        Args:
            env_idx: Environment index
            episode_num: Episode number
            num_robots: Number of robots in the episode
            timestep: Current training timestep
        """
        logger = self.get_logger()
        logger.info(
            f"Episode started - Env: {env_idx}, Episode: {episode_num}, "
            f"Robots: {num_robots}, Timestep: {timestep}"
        )
    
    def log_episode_end(self, env_idx, episode_num, episode_length, rewards, 
                        success_info, timestep):
        """
        Log episode completion event.
        
        Args:
            env_idx: Environment index
            episode_num: Episode number
            episode_length: Episode length in steps
            rewards: Dictionary or list of robot rewards
            success_info: Dictionary with success status for robots
            timestep: Current training timestep
        """
        logger = self.get_logger()
        
        # Format rewards
        if isinstance(rewards, dict):
            reward_str = ", ".join([f"R{i}: {r:.2f}" for i, r in rewards.items()])
        else:
            reward_str = ", ".join([f"R{i}: {r:.2f}" for i, r in enumerate(rewards)])
        
        logger.info(
            f"Episode ended - Env: {env_idx}, Episode: {episode_num}, "
            f"Length: {episode_length}, Timestep: {timestep}, "
            f"Rewards: [{reward_str}]"
        )
    
    def log_robot_deactivation(self, env_idx, robot_idx, reason, timestep, episode_step):
        """
        Log robot deactivation event.
        
        Args:
            env_idx: Environment index
            robot_idx: Robot index
            reason: Reason for deactivation ("collision" or "goal_reached")
            timestep: Current training timestep
            episode_step: Step within the episode
        """
        logger = self.get_logger()
        logger.info(
            f"Robot deactivated - Env: {env_idx}, Robot: {robot_idx}, "
            f"Reason: {reason}, Episode step: {episode_step}, Timestep: {timestep}"
        )
    
    def log_training_step(self, timestep, loss=None, exploration_rate=None):
        """
        Log training step metrics.
        
        Args:
            timestep: Current training timestep
            loss: Training loss value (optional)
            exploration_rate: Current exploration rate (optional)
        """
        logger = self.get_logger()
        msg_parts = [f"Training step - Timestep: {timestep}"]
        
        if loss is not None:
            msg_parts.append(f"Loss: {loss:.6f}")
        if exploration_rate is not None:
            msg_parts.append(f"Exploration: {exploration_rate:.4f}")
        
        logger.debug(", ".join(msg_parts))
    
    def log_evaluation(self, timestep, avg_reward, success_rate, avg_time=None, avg_energy=None):
        """
        Log evaluation results.
        
        Args:
            timestep: Current training timestep
            avg_reward: Average cumulative reward
            success_rate: Success rate (0-1)
            avg_time: Average time for successful episodes (optional)
            avg_energy: Average energy for successful episodes (optional)
        """
        logger = self.get_logger()
        msg_parts = [
            f"Evaluation - Timestep: {timestep}",
            f"Avg Reward: {avg_reward:.2f}",
            f"Success Rate: {success_rate:.2%}"
        ]
        
        if avg_time is not None:
            msg_parts.append(f"Avg Time: {avg_time:.2f}")
        if avg_energy is not None:
            msg_parts.append(f"Avg Energy: {avg_energy:.2f}")
        
        logger.info(", ".join(msg_parts))
    
    def log_checkpoint(self, timestep, checkpoint_path):
        """
        Log model checkpoint save event.
        
        Args:
            timestep: Current training timestep
            checkpoint_path: Path where checkpoint was saved
        """
        logger = self.get_logger()
        logger.info(f"Checkpoint saved - Timestep: {timestep}, Path: {checkpoint_path}")
    
    def log_timing(self, operation, duration, timestep=None):
        """
        Log timing information for operations.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            timestep: Current training timestep (optional)
        """
        logger = self.get_logger()
        msg = f"Timing - Operation: {operation}, Duration: {duration:.6f}s"
        if timestep is not None:
            msg += f", Timestep: {timestep}"
        logger.debug(msg)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
