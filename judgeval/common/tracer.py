"""
Tracing system for judgeval that allows for function tracing using decorators
and context managers.
"""

import time
import functools
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass

@dataclass
class TraceSpan:
    name: str
    start_time: float
    parent: Optional['TraceSpan'] = None
    depth: int = 0
    metadata: Optional[Dict] = None

class Tracer:
    """Singleton tracer class"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.current_span: Optional[TraceSpan] = None
            self.depth = 0
            self._enabled_count = 0  # Track number of active traces
            self.initialized = True

    @property
    def enabled(self) -> bool:
        return self._enabled_count > 0

    def observe(self, func=None, *, name=None, metadata=None):
        """
        Decorator that can be used with or without arguments:
        @tracer.observe
        def my_func(): ...
        
        or
        
        @tracer.observe(name="custom_name", metadata={...})
        def my_func(): ...
        """
        if func is None:
            # Called with arguments: @tracer.observe(name="...")
            return lambda f: self.observe(f, name=name, metadata=metadata)
        
        # Called without arguments: @tracer.observe
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._enabled_count += 1
            
            span_name = name or func.__name__
            span = TraceSpan(
                name=span_name,
                start_time=time.time(),
                parent=self.current_span,
                depth=self.depth,
                metadata=metadata
            )
            
            print(f"{'  ' * self.depth}→ {span.name}")
            self.current_span = span
            self.depth += 1
            
            try:
                result = func(*args, **kwargs)
                result_str = self._format_result(result, func.__name__)
                print(f"{'  ' * self.depth}{result_str}")
                return result
            finally:
                self.depth -= 1
                duration = time.time() - span.start_time
                print(f"{'  ' * self.depth}← {span.name} ({duration:.3f}s)")
                self.current_span = span.parent
                self._enabled_count -= 1
                
        return wrapper

    def span(self, name: str, metadata: Optional[Dict] = None):
        """Context manager for tracing code blocks"""
        class SpanContextManager:
            def __init__(self, tracer, name, metadata):
                self.tracer = tracer
                self.name = name
                self.metadata = metadata
                self.span = None

            def __enter__(self):
                if not self.tracer.enabled:
                    return self
                
                self.span = TraceSpan(
                    name=self.name,
                    start_time=time.time(),
                    parent=self.tracer.current_span,
                    depth=self.tracer.depth,
                    metadata=self.metadata
                )
                
                print(f"{'  ' * self.tracer.depth}→ {self.name}")
                self.tracer.current_span = self.span
                self.tracer.depth += 1
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if not self.tracer.enabled:
                    return
                
                self.tracer.depth -= 1
                duration = time.time() - self.span.start_time
                print(f"{'  ' * self.tracer.depth}← {self.name} ({duration:.3f}s)")
                self.tracer.current_span = self.span.parent

        return SpanContextManager(self, name, metadata)

    def _format_result(self, result: Any, func_name: str) -> str:
        """Format the result for display based on function and result type"""
        def get_scorer_info(scorer):
            """Helper function to get info regardless of type"""
            if isinstance(scorer, dict):
                return {
                    'name': scorer['name'],
                    'score': scorer['score'],
                    'success': scorer['success'],
                    'threshold': scorer.get('threshold', None),
                    'reason': scorer.get('reason', ''),
                    'evaluation_model': scorer.get('evaluation_model', None),
                    'verbose_logs': scorer.get('verbose_logs', ''),
                    'additional_metadata': scorer.get('additional_metadata', {})
                }
            else:  # ScorerData object
                return {
                    'name': scorer.name,
                    'score': scorer.score,
                    'success': scorer.success,
                    'threshold': getattr(scorer, 'threshold', None),
                    'reason': getattr(scorer, 'reason', ''),
                    'evaluation_model': getattr(scorer, 'evaluation_model', None),
                    'verbose_logs': getattr(scorer, 'verbose_logs', ''),
                    'additional_metadata': getattr(scorer, 'additional_metadata', {})
                }
        
        # Handle API evaluation results
        if func_name == "execute_api_eval":
            if isinstance(result, dict) and 'results' in result:
                summary = []
                for result_item in result['results']:
                    scorers_data = result_item['scorers_data']
                    for scorer in scorers_data:
                        summary.append(f"{'  ' * (self.depth + 1)}{scorer['name']}: {scorer['score']:.2f} ({scorer['success']})")
                return f"API Response with {len(result['results'])} results:\n" + "\n".join(summary)
        
        elif func_name == "run_eval":
            if isinstance(result, list) and len(result) > 0:
                output = ["Evaluation Results:"]
                
                for item in result:
                    # Add separator for readability
                    output.append("-" * 80)
                    
                    # Input/Output/Context
                    output.append(f"Q: {item.input}")
                    output.append(f"A: {item.actual_output}")
                    if item.expected_output:
                        output.append(f"Expected: {item.expected_output}")
                    if item.context:
                        output.append(f"Context: {item.context}")
                    if item.retrieval_context:
                        output.append(f"Retrieval Context: {item.retrieval_context}")
                    output.append("")  # Empty line for readability
                    
                    # Scorer results
                    for scorer_data in item.scorers_data:
                        info = get_scorer_info(scorer_data)
                        output.append(f"{info['name']}: {info['score']:.2f} (success={info['success']}, threshold={info['threshold']})")
                        
                        if info['reason']:
                            output.append(f"Reason: {info['reason']}")
                        
                        # Get claims and verdicts from additional_metadata
                        if 'claims' in info['additional_metadata']:
                            output.append("\nClaims:")
                            for claim in info['additional_metadata']['claims']:
                                output.append(f"• {claim['claim']}")
                                output.append(f"  Quote: {claim['quote']}")
                        
                        if 'verdicts' in info['additional_metadata']:
                            output.append("\nVerdicts:")
                            for verdict in info['additional_metadata']['verdicts']:
                                if isinstance(verdict, dict):
                                    output.append(f"• {verdict['verdict']}: {verdict['reason']}")
                                else:  # FaithfulnessVerdict object
                                    output.append(f"• {verdict.verdict}: {verdict.reason}")
                        
                        output.append(f"\nEvaluation Model: {info['evaluation_model']}")
                        output.append("")  # Empty line for readability
                
                return "\n".join(output)
            return "No evaluation results"
        
        else:
            return f"{func_name} has outputted {result}"
        
        


class Tracing:
    """Context manager for enabling tracing"""
    def __enter__(self):
        global tracer
        tracer.enabled = True
        return tracer

    def __exit__(self, exc_type, exc_val, exc_tb):
        global tracer
        tracer.enabled = False

# Create the global tracer instance at module level
tracer = Tracer()

# Optional: you can also expose it in __all__ to control what gets imported
__all__ = ['tracer', 'Tracing', 'TraceSpan']