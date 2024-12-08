"""
Tracing system for judgeval that allows for function tracing using decorators.
"""

import time
import functools
from typing import Optional, Any, List, Literal
from dataclasses import dataclass
import uuid

@dataclass
class TraceEntry:
    """
    Represents a single trace entry with its visual representation
    
    Each TraceEntry is a single line in the trace.
    The `type` field determines the visual representation of the entry.
    - `enter` is for when a function is entered, represented by `→`
    - `exit` is for when a function is exited, represented by `←`
    - `output` is for when a function outputs a value, represented by `Output:`

    Args:
        type: The type of trace entry ('enter', 'exit', or 'output')
        function: Name of the function being traced
        depth: Indentation level of this trace entry
        message: Additional message to include in the trace
        timestamp: Time when this trace entry was created
        duration: For 'exit' entries, how long the function took to execute
        output: For 'output' entries, the value that was output
    """
    type: Literal['enter', 'exit', 'output']
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

class TraceClient:
    """Client for managing a single trace context"""
    def __init__(self, tracer, trace_id: str, name: str):
        self.tracer = tracer
        self.trace_id = trace_id
        self.name = name
        self.entries: List[TraceEntry] = []
        self.start_time = time.time()
        
    def add_entry(self, entry: TraceEntry):
        """Add a trace entry to this trace context"""
        self.entries.append(entry)
        return self
        
    def print_trace(self):
        """Print the complete trace with proper visual structure"""
        for entry in self.entries:
            entry.print_entry()
            
    def get_duration(self) -> float:
        """Get the total duration of this trace"""
        return time.time() - self.start_time

class Tracer:
    """
    A singleton tracer class that provides function execution tracing capabilities.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.depth = 0
            self._current_trace: Optional[TraceClient] = None
            self.initialized = True
            
    def start_trace(self, name: str = None) -> TraceClient:
        """Start a new trace context"""
        trace_id = str(uuid.uuid4())
        self._current_trace = TraceClient(self, trace_id, name or "unnamed_trace")
        return self._current_trace

    def observe(self, func=None, *, name=None):
        """
        Decorator to trace function execution with detailed entry/exit information.
        
        Args:
            func: The function to trace
            name: Optional custom name for the function
        """
        if func is None:
            return lambda f: self.observe(f, name=name)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self._current_trace:
                span_name = name or func.__name__
                start_time = time.time()
                
                # Record function entry
                self._current_trace.add_entry(TraceEntry(
                    type="enter",
                    function=span_name,
                    depth=self.depth,
                    message=f"→ {span_name}",
                    timestamp=start_time
                ))
                
                self.depth += 1
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record the output
                    self._current_trace.add_entry(TraceEntry(
                        type="output",
                        function=span_name,
                        depth=self.depth,
                        message=f"Output from {span_name}",
                        timestamp=time.time(),
                        output=result
                    ))
                    
                    return result
                    
                finally:
                    self.depth -= 1
                    duration = time.time() - start_time
                    
                    # Record function exit
                    self._current_trace.add_entry(TraceEntry(
                        type="exit",
                        function=span_name,
                        depth=self.depth,
                        message=f"← {span_name}",
                        timestamp=time.time(),
                        duration=duration
                    ))
            
            return func(*args, **kwargs)
            
        return wrapper

# Create the global tracer instance
tracer = Tracer()

# Export only what's needed
__all__ = ['tracer']