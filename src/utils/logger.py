"""
============================================================================
Custom Logging System
Provides colored console output and file logging with timestamps
============================================================================
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Try to import colorama for colored output (optional)
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback: no colors
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        if COLORAMA_AVAILABLE:
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
                record.msg = f"{self.COLORS[levelname]}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


class CustomLogger:
    """
    Custom logger with console and file output
    
    Features:
    - Colored console output (if colorama available)
    - File logging with timestamps
    - Multiple log levels
    - Progress tracking
    """
    
    def __init__(self, name: str = "WaterContamination", log_dir: Optional[Path] = None):
        """
        Initialize custom logger
        
        Args:
            name: Logger name
            log_dir: Directory to save log files (optional)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatters
        console_formatter = ColoredFormatter(
            '%(levelname)s - %(message)s'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_dir provided)
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"execution_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        else:
            self.log_file = None
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def section(self, title: str, width: int = 70):
        """
        Log a section header
        
        Args:
            title: Section title
            width: Width of the separator line
        """
        separator = "=" * width
        self.logger.info(separator)
        self.logger.info(title.center(width))
        self.logger.info(separator)
    
    def subsection(self, title: str, width: int = 70):
        """
        Log a subsection header
        
        Args:
            title: Subsection title
            width: Width of the separator line
        """
        separator = "-" * width
        self.logger.info(separator)
        self.logger.info(title)
        self.logger.info(separator)
    
    def progress(self, current: int, total: int, prefix: str = "Progress"):
        """
        Log progress
        
        Args:
            current: Current step
            total: Total steps
            prefix: Progress message prefix
        """
        percentage = (current / total) * 100
        self.logger.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")
    
    def success(self, message: str):
        """Log success message (green)"""
        if COLORAMA_AVAILABLE:
            self.logger.info(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
        else:
            self.logger.info(f"✓ {message}")
    
    def failure(self, message: str):
        """Log failure message (red)"""
        if COLORAMA_AVAILABLE:
            self.logger.error(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")
        else:
            self.logger.error(f"✗ {message}")
    
    def step(self, step_num: int, total_steps: int, message: str):
        """
        Log a step in a process
        
        Args:
            step_num: Current step number
            total_steps: Total number of steps
            message: Step description
        """
        self.logger.info(f"[{step_num}/{total_steps}] {message}")
    
    def metric(self, name: str, value: float, unit: str = ""):
        """
        Log a metric value
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement (optional)
        """
        if unit:
            self.logger.info(f"{name}: {value:.4f} {unit}")
        else:
            self.logger.info(f"{name}: {value:.4f}")
    
    def table(self, data: dict, title: Optional[str] = None):
        """
        Log data in table format
        
        Args:
            data: Dictionary of key-value pairs
            title: Optional table title
        """
        if title:
            self.logger.info(f"\n{title}:")
        
        # Calculate max key length for alignment
        max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
        
        for key, value in data.items():
            self.logger.info(f"  {str(key):<{max_key_len}} : {value}")
    
    def timer_start(self, task_name: str):
        """Start timing a task"""
        self.logger.info(f"Starting: {task_name}...")
        return datetime.now()
    
    def timer_end(self, task_name: str, start_time: datetime):
        """End timing a task"""
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Completed: {task_name} ({duration:.2f}s)")
        return duration
    
    def exception(self, message: str, exc_info: bool = True):
        """Log exception with traceback"""
        self.logger.exception(message, exc_info=exc_info)


# Global logger instance
_global_logger: Optional[CustomLogger] = None


def get_logger(name: str = "WaterContamination", log_dir: Optional[Path] = None) -> CustomLogger:
    """
    Get or create global logger instance
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
    
    Returns:
        CustomLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = CustomLogger(name=name, log_dir=log_dir)
    
    return _global_logger


def reset_logger():
    """Reset global logger (useful for testing)"""
    global _global_logger
    _global_logger = None


# Convenience functions for quick logging
def info(message: str):
    """Quick info log"""
    logger = get_logger()
    logger.info(message)


def error(message: str):
    """Quick error log"""
    logger = get_logger()
    logger.error(message)


def success(message: str):
    """Quick success log"""
    logger = get_logger()
    logger.success(message)


def warning(message: str):
    """Quick warning log"""
    logger = get_logger()
    logger.warning(message)


if __name__ == "__main__":
    # Test the logger
    import tempfile
    
    # Create test logger with temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = CustomLogger(name="TestLogger", log_dir=Path(temp_dir))
        
        logger.section("Logger Test")
        logger.info("This is an info message")
        logger.success("This is a success message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        
        logger.subsection("Progress Test")
        for i in range(1, 6):
            logger.progress(i, 5, prefix="Testing")
        
        logger.subsection("Metrics Test")
        logger.metric("Accuracy", 0.8542, "%")
        logger.metric("Loss", 0.1234)
        
        logger.subsection("Table Test")
        logger.table({
            "Model": "Random Forest",
            "Accuracy": "85.42%",
            "F1-Score": "0.8234",
            "Training Time": "5.2s"
        }, title="Model Performance")
        
        logger.subsection("Timer Test")
        import time
        start = logger.timer_start("Test Task")
        time.sleep(0.5)
        logger.timer_end("Test Task", start)
        
        print(f"\n✓ Log file created: {logger.log_file}")
