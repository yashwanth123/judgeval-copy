from judgeval.tracer import Tracer
from .processor import DataProcessor

judgment = Tracer(project_name="multi_file_demo")

@judgment.observe(span_type="function")
def main():
    """Main function demonstrating multi-file tracing."""
    judgment.log("Starting multi-file demo")
    
    processor = DataProcessor()
    
    valid_data = [1, 2, 3, 4, 5]
    judgment.log("Processing valid data batch")
    result = processor.process_batch(valid_data)
    judgment.log(f"Valid data processing result: {result}")
    
    invalid_data = [1, -2, 3, 4, 5]
    judgment.log("Processing invalid data batch")
    result = processor.process_batch(invalid_data)
    judgment.log(f"Invalid data processing result: {result}")
    
    stats = processor.get_statistics()
    judgment.log(f"Overall statistics: {stats}")
    
    judgment.log("Multi-file demo completed")

if __name__ == "__main__":
    main() 