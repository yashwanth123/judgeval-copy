"""
Tracing system for judgeval that allows for function tracing using decorators.
"""

import time
import functools
from typing import Optional, Any, List, Literal
from dataclasses import dataclass

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

@dataclass
class TracedResult:
    """
    Stores the result and trace information for a single traced function execution.

    This class encapsulates both the return value and the execution trace of a traced function.
    It provides methods to visualize the complete execution trace.

    Args:
        result (Any): The value returned by the traced function
        trace_entries (List[TraceEntry]): List of TraceEntry objects representing the execution trace
        name (str): Name of the traced function

    Example:
        traced = TracedResult(
            result=42,
            trace_entries=[entry1, entry2],
            name="my_function"
        )
        traced.print_trace()  # Prints the complete execution trace
    """
    result: Any
    trace_entries: List[TraceEntry]
    name: str

    def print_trace(self):
        """
        Print the complete trace with proper visual structure.
        
        Iterates through all trace entries and prints them with appropriate
        indentation and visual markers (→, ←, Output:) to show the execution flow.
        Each entry is printed according to its type:
        - Function entry (→)
        - Function exit (←) with duration
        - Function output with the output value
        """
        for entry in self.trace_entries:
            entry.print_entry()

class Tracer:
    """
    A singleton tracer class that provides function execution tracing capabilities.

    This class implements the singleton pattern and provides functionality to trace function
    executions, including their entry, exit, duration, and output values. It maintains a
    hierarchical trace of function calls with proper depth tracking.

    Attributes:
        depth (int): Current depth of function call nesting
        _is_tracing (bool): Flag indicating if tracing is currently active
        _current_trace_outputs (list): List of TraceEntry objects for the current trace
        initialized (bool): Flag indicating if the singleton has been initialized
    """
    _instance = None

    def __new__(cls):
        """Ensure only one instance of Tracer exists (singleton pattern).

        Returns:
            Tracer: The single instance of the Tracer class
        """
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
        """
        Decorator to trace function execution with detailed entry/exit information.

        This method can be used as a decorator with or without parameters. It tracks function
        entry, exit, duration, and output values, maintaining a hierarchical trace structure.

        Args:
            func (callable, optional): The function to be traced
            name (str, optional): Custom name for the traced function
            metadata (dict, optional): Additional metadata for the trace
            top_level (bool, optional): If True, starts a new trace context

        Returns:
            callable: Decorated function that includes tracing functionality
            
        Example:
            @tracer.observe(name="custom_name")
            def my_function():
                pass
        """
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