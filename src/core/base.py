"""
Base classes for services.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
import threading
import time
import logging


@dataclass
class ServiceResult:
    """Result from a service operation."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class BaseService(ABC):
    """Base class for all services."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or self._create_logger()
        self._initialized = False
        # Guards one-time model loading so concurrent callers (e.g. multiple
        # batch workers sharing one service) don't run initialize() at the
        # same time. The first caller loads; the rest wait, then reuse it.
        self._init_lock = threading.Lock()
    
    def _create_logger(self) -> logging.Logger:
        """Create logger for this service."""
        logger = logging.getLogger(f"service.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            # Don't bubble up to root — root may have its own handler
            # (e.g. uvicorn / basicConfig), which would emit each record twice.
            logger.propagate = False
        return logger
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the service (load models, etc.)."""
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> ServiceResult:
        """Process input and return result."""
        pass
    
    def _execute_with_timing(
        self,
        operation: callable,
        *args,
        **kwargs
    ) -> ServiceResult:
        """Execute operation with timing and error handling."""
        start_time = time.time()
        
        try:
            result_data = operation(*args, **kwargs)
            execution_time = time.time() - start_time
            
            self.logger.info(
                f"Service {self.name} completed in {execution_time:.2f}s"
            )
            
            return ServiceResult(
                success=True,
                data=result_data,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(
                f"Service {self.name} failed after {execution_time:.2f}s: {error_msg}"
            )
            
            return ServiceResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    def ensure_initialized(self) -> None:
        """Ensure the service is initialized exactly once, thread-safely.

        Double-checked locking: the common already-initialized path takes no
        lock (zero overhead per process() call); the first concurrent callers
        serialize on `_init_lock` so initialize() runs once and a second
        worker waits for the load instead of racing a half-built model.
        """
        if self._initialized:
            return
        with self._init_lock:
            if not self._initialized:
                self.initialize()
                self._initialized = True

