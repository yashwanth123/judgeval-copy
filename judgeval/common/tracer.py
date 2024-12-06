"""
Tracing system for judgeval that allows for function tracing using decorators.
"""

import time
import functools
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class TraceEntry:
    """Represents a single trace entry with its visual representation"""
    type: str  # 'enter', 'exit', or 'output'
    function: str
    depth: int
    message: str
    timestamp: float
    duration: Optional[float] = None
    output: Any = None

    def print_entry(self):
        indent = "  " * self.depth
        if self.type == "enter":
            print(f"{indent}→ {self.function}")
        elif self.type == "exit":
            print(f"{indent}← {self.function} ({self.duration:.3f}s)")
        elif self.type == "output":
            print(f"{indent}Output: {self.output}")

@dataclass
class TracedResult:
    """Stores the result and trace information for a single traced function execution"""
    result: Any
    trace_entries: List[TraceEntry]
    name: str

    def print_trace(self):
        """Print the complete trace with proper visual structure"""
        for entry in self.trace_entries:
            entry.print_entry()

class Tracer:
    """Singleton tracer class"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.depth = 0
            self._is_tracing = False
            self._current_trace_outputs = []
            self.initialized = True

    def observe(self, func=None, *, name=None, metadata=None, top_level=False):
        if func is None:
            return lambda f: self.observe(f, name=name, metadata=metadata, top_level=top_level)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            should_trace = self._is_tracing or top_level
            
            if top_level:
                self._is_tracing = True
                self._current_trace_outputs = []
            
            if should_trace:
                span_name = name or func.__name__
                start_time = time.time()
                
                # Record function entry
                self._current_trace_outputs.append(TraceEntry(
                    type="enter",
                    function=span_name,
                    depth=self.depth,
                    message=f"→ {span_name}",
                    timestamp=start_time
                ))
                
                self.depth += 1
            
            try:
                result = func(*args, **kwargs)
                if should_trace:
                    current_time = time.time()
                    duration = current_time - start_time
                    
                    # Record the output
                    self._current_trace_outputs.append(TraceEntry(
                        type="output",
                        function=span_name,
                        depth=self.depth,
                        message=f"Output from {span_name}",
                        timestamp=current_time,
                        output=result
                    ))
                    
                    # Record the result timing
                    self._current_trace_outputs.append(TraceEntry(
                        type="result",
                        function=span_name,
                        depth=self.depth,
                        message=f"= {span_name} result",
                        timestamp=current_time,
                        duration=duration
                    ))
                
                if top_level:
                    traced_result = TracedResult(
                        result=result,
                        trace_entries=self._current_trace_outputs.copy(),
                        name=span_name
                    )
                    return traced_result
                return result
                
            finally:
                if should_trace:
                    self.depth -= 1
                    duration = time.time() - start_time
                    
                    # Record function exit
                    self._current_trace_outputs.append(TraceEntry(
                        type="exit",
                        function=span_name,
                        depth=self.depth,
                        message=f"← {span_name}",
                        timestamp=time.time(),
                        duration=duration
                    ))
                
                if top_level:
                    self._is_tracing = False
        
        return wrapper

# Create the global tracer instance
tracer = Tracer()

# Export only what's needed
__all__ = ['tracer']