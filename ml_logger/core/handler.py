"""Logging handler that feeds formatted records to the live dashboard."""

import logging
from collections import deque
from threading import Lock

class DashboardLogHandler(logging.Handler):
    """Custom logging handler that routes formatted logs to a memory deque."""
    def __init__(self, log_queue: deque, queue_lock: Lock, highlighter):
        super().__init__()
        self.log_queue = log_queue
        self.queue_lock = queue_lock
        self.highlighter = highlighter

    def emit(self, record):
        """Format, highlight, and append one record under the queue lock."""
        try:
            msg = self.format(record)
            
            # Apply dynamic rich highlighting
            text_obj = self.highlighter.apply(msg)
            
            # Append to live dashboard memory
            with self.queue_lock:
                self.log_queue.append(text_obj)
        except Exception:
            self.handleError(record)
