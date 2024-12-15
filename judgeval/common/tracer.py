"""
Tracing system for judgeval that allows for function tracing using decorators.
"""

import time
import functools
import requests
import uuid

from typing import Optional, Any, List, Literal
from dataclasses import dataclass
from datetime import datetime 

from judgeval.constants import JUDGMENT_TRACES_SAVE_API_URL

@dataclass
class TraceEntry:
    """
    Represents a single trace entry with its visual representation
    
    Each TraceEntry is a single line in the trace.
    The `type` field determines the visual representation of the entry.
        - `enter` is for when a function is entered, represented by `→`
        - `exit` is for when a function is exited, represented by `←`
        - `output` is for when a function outputs a value, represented by `Output:`
        - `input` is for function input parameters, represented by `Input:`

    function: Name of the function being traced
    depth: Indentation level of this trace entry
    message: Additional message to include in the trace
    timestamp: Time when this trace entry was created
    duration: For 'exit' entries, how long the function took to execute
    output: For 'output' entries, the value that was output
    """
    type: Literal['enter', 'exit', 'output', 'input']
    function: str
    depth: int
    message: str
    timestamp: float
    duration: Optional[float] = None
    output: Any = None
    inputs: dict = None

    def print_entry(self):
        indent = "  " * self.depth
        if self.type == "enter":
            print(f"{indent}→ {self.function}")
        elif self.type == "exit":
            print(f"{indent}← {self.function} ({self.duration:.3f}s)")
        elif self.type == "output":
            print(f"{indent}Output: {self.output}")
        elif self.type == "input":
            print(f"{indent}Input: {self.inputs}")
    
    def to_dict(self):
        """Convert the trace entry to a dictionary format"""
        return {
            "type": self.type,
            "function": self.function,
            "depth": self.depth,
            "message": self.message,
            "timestamp": self.timestamp,
            "duration": self.duration,
            # TODO: converting these to strings may be a problem
            "output": str(self.output),  # Convert output to string to ensure JSON serializable
            "inputs": str(self.inputs) if self.inputs else None  # Convert inputs to string
        }

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
        """
        Get the total duration of this trace
        """
        return time.time() - self.start_time
    
    def condense_trace(self, entries: List[dict]) -> List[dict]:
        """
        Condenses trace entries into a single entry for each function.
        
        Groups entries by function call and combines them into a single entry with:
        - depth: deepest depth for this function call
        - duration: time from first to last timestamp 
        - function: function name
        - inputs: non-None inputs
        - output: non-None outputs
        - timestamp: first timestamp of the function call
        """
        condensed = []
        current_func = None
        current_entry = None

        for entry in entries:
            if entry["type"] == "enter":
                # Start of new function call
                current_func = entry["function"]
                current_entry = {
                    "depth": entry["depth"],
                    "function": entry["function"],
                    "timestamp": entry["timestamp"],
                    "inputs": None,
                    "output": None
                }
            
            elif entry["type"] == "exit" and entry["function"] == current_func:
                # End of current function
                current_entry["duration"] = entry["timestamp"] - current_entry["timestamp"]
                condensed.append(current_entry)
                current_func = None
                current_entry = None
            
            elif current_func and entry["function"] == current_func:
                # Additional entries for current function
                if entry["depth"] > current_entry["depth"]:
                    current_entry["depth"] = entry["depth"]
                
                if entry["type"] == "input" and entry["inputs"]:
                    current_entry["inputs"] = entry["inputs"]
                    
                if entry["type"] == "output" and entry["output"]:
                    current_entry["output"] = entry["output"]

        return condensed

    def save_trace(self) -> dict:
        """
        Save the current trace to the database.
        Returns the trace data that was saved.
        """
        # Calculate total elapsed time
        total_duration = self.get_duration()
        
        raw_entries = [entry.to_dict() for entry in self.entries]
        condensed_entries = self.condense_trace(raw_entries)

        # Create trace document
        trace_data = {
            "trace_id": self.trace_id,
            "api_key": self.tracer.api_key,
            "name": self.name,
            "created_at": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration": total_duration,
            "token_counts": {
                "prompt_tokens": 0,  # Dummy value
                "completion_tokens": 0,  # Dummy value
                "total_tokens": 0,  # Dummy value
            },  # TODO: Add token counts
            "entries": condensed_entries
        }

        # Save trace data by making POST request to API
        response = requests.post(
            JUDGMENT_TRACES_SAVE_API_URL,
            json=trace_data,
            headers={
                "Content-Type": "application/json",
            }
        )
        response.raise_for_status()
        return trace_data

class Tracer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.depth = 0
            self._current_trace: Optional[TraceClient] = None
            self.api_key: Optional[str] = None
            self.initialized = True
    
    def configure(self, api_key: str):
        """Configure the tracer with an API key"""
        self.api_key = api_key
        
    def start_trace(self, name: str = None) -> TraceClient:
        """Start a new trace context"""
        if not self.api_key:
            raise ValueError("Tracer must be configured with an API key first")
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
                
                # Record function inputs
                inputs = {
                    'args': args,
                    'kwargs': kwargs
                }
                self._current_trace.add_entry(TraceEntry(
                    type="input",
                    function=span_name,
                    depth=self.depth + 1,  # Indent inputs under function entry
                    message=f"Inputs to {span_name}",
                    timestamp=time.time(),
                    inputs=inputs
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