"""
Tracing system for judgeval that allows for function tracing using decorators
and context managers.
"""

import time
import uuid
import functools
from typing import Optional, Any, Callable, Dict
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class TraceSpan:
    name: str
    start_time: float
    parent: Optional['TraceSpan'] = None
    depth: int = 0
    metadata: Optional[Dict] = None

class Tracer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.current_span: Optional[TraceSpan] = None
            self.depth = 0
            self.enabled = False
            self.initialized = True

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict] = None):
        """Create a new tracing span"""
        if not self.enabled:
            yield None
            return

        span = TraceSpan(
            name=name,
            start_time=time.time(),
            parent=self.current_span,
            depth=self.depth,
            metadata=metadata
        )
        
        self._enter_span(span)
        try:
            yield span
        finally:
            self._exit_span(span)

    def observe(self, name: Optional[str] = None, metadata: Optional[Dict] = None) -> Callable:
        """Decorator to trace function execution"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                span_name = name or func.__name__
                with self.span(span_name, metadata):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def _enter_span(self, span: TraceSpan):
        print(f"{'  ' * self.depth}→ {span.name}")
        self.current_span = span
        self.depth += 1

    def _exit_span(self, span: TraceSpan):
        self.depth -= 1
        duration = time.time() - span.start_time
        print(f"{'  ' * self.depth}← {span.name} ({duration:.3f}s)")
        self.current_span = span.parent

class Tracing:
    """Context manager for enabling/disabling tracing"""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.tracer = Tracer()
        self.previous_enabled = None

    def __enter__(self):
        self.previous_enabled = self.tracer.enabled
        self.tracer.enabled = self.enabled
        return self.tracer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracer.enabled = self.previous_enabled

# Create global tracer instance
tracer = Tracer()