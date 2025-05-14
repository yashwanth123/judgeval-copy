from typing import List, Dict, Optional
from judgeval.tracer import Tracer
from .utils import process_data, validate_input

judgment = Tracer(project_name="multi_file_demo")

class DataProcessor:
    def __init__(self):
        self.processed_data: List[Dict] = []
    
    def process_batch(self, data: List[int]) -> Optional[Dict]:
        """Process a batch of numbers."""
        judgment.log("Starting batch processing")
        
        if not validate_input(data):
            judgment.log("Input validation failed", label="error")
            return None
        
        result = process_data(data)
        self.processed_data.append(result)
        
        judgment.log(f"Successfully processed batch with {result['count']} numbers")
        return result
    
    def get_statistics(self) -> Dict:
        """Get statistics across all processed batches."""
        judgment.log("Calculating overall statistics")
        
        if not self.processed_data:
            judgment.log("No data has been processed yet", level="warning")
            return {"total_batches": 0, "total_numbers": 0}
        
        total_batches = len(self.processed_data)
        total_numbers = sum(batch["count"] for batch in self.processed_data)
        
        judgment.log(f"Processed {total_batches} batches with {total_numbers} total numbers")
        return {
            "total_batches": total_batches,
            "total_numbers": total_numbers
        } 