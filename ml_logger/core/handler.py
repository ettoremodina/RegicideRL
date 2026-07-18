import logging
from collections import deque
from threading import Lock

class DashboardLogHandler(logging.Handler):
    """Custom logging handler that routes formatted logs to a memory deque and a file writer."""
    def __init__(self, log_queue: deque, queue_lock: Lock, highlighter, file_writer):
        super().__init__()
        self.log_queue = log_queue
        self.queue_lock = queue_lock
        self.highlighter = highlighter
        self.file_writer = file_writer

    def emit(self, record):
        try:
            msg = self.format(record)
            
            # Apply dynamic rich highlighting
            text_obj = self.highlighter.apply(msg)
            
            # Save to disk
            self.file_writer.log_message(text_obj)
            
            # Append to live dashboard memory
            with self.queue_lock:
                self.log_queue.append(text_obj)
        except Exception:
            self.handleError(record)
