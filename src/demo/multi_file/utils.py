from typing import List, Dict
from judgeval.tracer import Tracer

judgment = Tracer(project_name="multi_file_demo")

def process_data(data: List[int]) -> Dict[str, int]:
    """Process a list of numbers and return statistics."""
    judgment.log("Starting data processing")
    
    if not data:
        judgment.log("Empty data received", level="warning")
        return {"count": 0, "sum": 0, "average": 0}
    
    count = len(data)
    total = sum(data)
    average = total / count
    
    judgment.log(f"Processed {count} numbers")
    return {
        "count": count,
        "sum": total,
        "average": average
    }

def validate_input(data: List[int]) -> bool:
    """Validate that all numbers are positive."""
    judgment.log("Validating input data")
    
    for num in data:
        if num < 0:
            judgment.log(f"Found negative number: {num}", label="error")
            return False
    
    judgment.log("All numbers are positive")
    return True 