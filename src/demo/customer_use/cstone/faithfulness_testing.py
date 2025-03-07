import os
from typing import List
import csv
import json
import anthropic
from openai import OpenAI
import openai
import time
import asyncio
from langsmith import wrappers
from langsmith import Client as langsmith_client 
import pandas as pd
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer
from dotenv import load_dotenv
from patronus import Client as PatronusClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_data(
    file_path: str, 
    docket_id_column: str, 
    actual_output_column: str, 
    retrieval_context_column: str,
    label_column: str,
    flip_labels: bool = False,
    sampling_limit: int = None
    ):
    """Load and parse the data from CSV"""
    examples, gold_labels = [], []

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header row
        # Get column indices from header
        docket_id_idx = header.index(docket_id_column)
        actual_output_idx = header.index(actual_output_column)
        retrieval_context_idx = header.index(retrieval_context_column)
        label_idx = header.index(label_column)
        # Read the data
        data = list(reader)
        
        # Apply sampling limit if specified
        if sampling_limit is not None and sampling_limit < len(data):
            import random
            data = random.sample(data, sampling_limit)
        
    for row in data:
        example = Example(
            input=str(row[docket_id_idx]),
            actual_output=row[actual_output_idx],
            retrieval_context=[row[retrieval_context_idx]],
        )
        examples.append(example)

        orig_label = row[label_idx]
        if flip_labels:
            gold_labels.append(not orig_label)
        else:
            gold_labels.append(orig_label)

    print(f"Loaded {len(examples)} examples from dataset.")
        
    return examples, gold_labels

def run_judgment_evaluation(examples: List[Example], model: str, run_threshold: float = 1.0) -> List[bool]:
    """
    Run evaluation using JudgmentClient
    
    Args:
        examples: List of Example objects
        
    Returns:
        List of boolean values indicating if the example hallucinated
    """
    client = JudgmentClient()
    scorer = FaithfulnessScorer(threshold=1.0)
    
    output = client.run_evaluation(
        model=model,
        examples=examples,
        scorers=[scorer],
        eval_run_name=f"cstone-basic-test-{model}-{run_threshold}", 
        project_name="cstone_faithfulness_testing",
        override=True,
    )
    
    scores = []
    for result in output:
        score = result.scorers_data[0].score
        scores.append(score)

    print(f"Judgment Scores: {scores}")
    # score < threshold ==> hallucination. score >= threshold ==> not hallucination
    return [score < run_threshold for score in scores]

def run_base_evaluation(examples: List[Example], model: str) -> List[bool]:
    """
    Run evaluation using base approach
    """
    import concurrent.futures
    
    BASE_HALLUCINATION_PROMPT = """
    You are a helpful assistant that determines if a given text is a hallucination.
    A hallucination is defined as a statement in the model output that is not supported by the provided context.

    You will be given a model output and a retrieval context. You will need to determine if the model output is a hallucination.

    Please respond with "True" if the model output is a hallucination, and "False" if it is not.
    ONLY RESPOND WITH "True" OR "False". DO NOT RESPOND WITH ANY OTHER TEXT.
    """
    
    def process_example(ex):
        actual_output = ex.actual_output
        retrieval_context = ex.retrieval_context
        
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": BASE_HALLUCINATION_PROMPT},
                {"role": "user", "content": f"Model Output: {actual_output}\nRetrieval Context: {retrieval_context}"}
            ]
        ).choices[0].message.content
        
        if response == "True":
            return True
        elif response == "False":
            return False
        else:
            raise ValueError(f"Invalid response: {response}")
    
    # Use ThreadPoolExecutor to parallelize API calls
    with concurrent.futures.ThreadPoolExecutor() as executor:
        predictions = list(executor.map(process_example, examples))
    
    return predictions

def run_patronus_evaluation(examples: List[Example]):
    """
    Run evaluation using PatronusClient
    
    Args:
        examples: List of Example objects
        
    Returns:
        List of boolean values indicating if the example is a false negative
    """
    patronus_client = PatronusClient(api_key=os.getenv("PATRONUS_API_KEY"))
    scores = []
    
    for example in examples:
        result = patronus_client.evaluate(
            evaluator="lynx-small",
            criteria="patronus:hallucination",
            evaluated_model_input="Does the context identify a class action lawsuit?",
            evaluated_model_output=example.actual_output,
            evaluated_model_retrieved_context=example.retrieval_context,
            tags={"scenario": "cstone"},
        )
        scores.append(result.score_raw)

    print(f"patronus scores: {scores}")
        
    return [score < 0.9 for score in scores]

def evaluate_predictions(docket_ids: List[str], predictions: List[bool], gold_labels: List[bool]):
    """Calculate metrics comparing predictions to gold labels"""
    
    # Find false negatives
    false_negatives = [docket_id for docket_id, gold, pred in zip(docket_ids, gold_labels, predictions) if gold and not pred]
    
    # Calculate confusion matrix metrics
    # is_hallucinated is the gold label, predictions is a bool where True = hallucinated, False = not hallucinated
    TP = sum(1 for g, p in zip(gold_labels, predictions) if g and p)
    FP = sum(1 for g, p in zip(gold_labels, predictions) if not g and p) 
    TN = sum(1 for g, p in zip(gold_labels, predictions) if not g and not p)
    FN = sum(1 for g, p in zip(gold_labels, predictions) if g and not p)
    
    # Calculate metrics for hallucinated class
    if TP + FP == 0:
        precision_hall = 0
    else:
        precision_hall = TP / (TP + FP)
        
    if TP + FN == 0:
        recall_hall = 0
    else:
        recall_hall = TP / (TP + FN)
        
    if precision_hall + recall_hall == 0:
        f1_hall = 0
    else:
        f1_hall = 2 * (precision_hall * recall_hall) / (precision_hall + recall_hall)
    
    # Calculate metrics for not-hallucinated class
    if TN + FN == 0:
        precision_not_hall = 0
    else:
        precision_not_hall = TN / (TN + FN)
        
    if TN + FP == 0:
        recall_not_hall = 0
    else:
        recall_not_hall = TN / (TN + FP)
        
    if precision_not_hall + recall_not_hall == 0:
        f1_not_hall = 0
    else:
        f1_not_hall = 2 * (precision_not_hall * recall_not_hall) / (precision_not_hall + recall_not_hall)
    
    # Calculate macro average (unweighted mean of both classes)
    precision_macro = (precision_hall + precision_not_hall) / 2
    recall_macro = (recall_hall + recall_not_hall) / 2
    f1_macro = (f1_hall + f1_not_hall) / 2
    
    # Calculate weighted average based on class support
    total = TP + FP + TN + FN
    support_hall = TP + FN
    support_not_hall = TN + FP
    
    precision_weighted = ((precision_hall * support_hall) + (precision_not_hall * support_not_hall)) / total
    recall_weighted = ((recall_hall * support_hall) + (recall_not_hall * support_not_hall)) / total
    f1_weighted = ((f1_hall * support_hall) + (f1_not_hall * support_not_hall)) / total
    
    return {
        # Original metrics (hallucinated class)
        'precision': precision_hall,
        'recall': recall_hall,
        'f1': f1_hall,
        'false_negatives': false_negatives,
        
        # Added metrics for both classes
        'class_metrics': {
            'hallucinated': {
                'precision': precision_hall,
                'recall': recall_hall,
                'f1': f1_hall,
                'support': support_hall
            },
            'not_hallucinated': {
                'precision': precision_not_hall,
                'recall': recall_not_hall,
                'f1': f1_not_hall,
                'support': support_not_hall
            }
        },
        
        # Averages across classes
        'macro_avg': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'weighted_avg': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        }
    }


if __name__ == "__main__":
    # Load data
    FILE_PATH = "/Users/alexshan/Desktop/judgment_labs/judgeval/src/demo/customer_use/cstone/JudgmentDemo/clh-ma-class-action-sec-v3.csv"
    
    examples, gold_labels = load_data(
        FILE_PATH, 
        docket_id_column="docket_id", 
        actual_output_column="LLM_raw_response", 
        retrieval_context_column="excerpts",
        label_column="correct",
        flip_labels=True,
        sampling_limit=10
    )
    
    docket_ids = [example.input for example in examples]
    print(f"docket_ids: {docket_ids}")

    # Run evaluations
    judgment_predictions = run_judgment_evaluation(
        examples, 
        model="osiris-mini", 
        run_threshold=0.95
    )

    # base_predictions = run_base_evaluation(examples, model="gpt-4o")
    # judgment_predictions = base_predictions
    # patronus_predictions = run_patronus_evaluation(examples)

    print(f"Judgment Predictions: {judgment_predictions}")
    print(f"Gold Labels: {gold_labels}")
    # Patronus Predictions: {patronus_predictions}")

    # Calculate and print metrics for both
    judgment_metrics = evaluate_predictions(
        docket_ids=docket_ids, 
        predictions=judgment_predictions,   # whether the model output is a hallucination
        gold_labels=gold_labels
    )
    # patronus_metrics = evaluate_predictions(patronus_predictions)
    
    print(f"\nMetrics for Judgment:")
    print(f"Class: Hallucinated")
    print(f"  Precision: {judgment_metrics['class_metrics']['hallucinated']['precision']:.4f}")
    print(f"  Recall: {judgment_metrics['class_metrics']['hallucinated']['recall']:.4f}")
    print(f"  F1 Score: {judgment_metrics['class_metrics']['hallucinated']['f1']:.4f}")
    print(f"  Support: {judgment_metrics['class_metrics']['hallucinated']['support']}")

    print(f"\nClass: Not Hallucinated")
    print(f"  Precision: {judgment_metrics['class_metrics']['not_hallucinated']['precision']:.4f}")
    print(f"  Recall: {judgment_metrics['class_metrics']['not_hallucinated']['recall']:.4f}")
    print(f"  F1 Score: {judgment_metrics['class_metrics']['not_hallucinated']['f1']:.4f}")
    print(f"  Support: {judgment_metrics['class_metrics']['not_hallucinated']['support']}")

    print(f"\nMacro Average:")
    print(f"  Precision: {judgment_metrics['macro_avg']['precision']:.4f}")
    print(f"  Recall: {judgment_metrics['macro_avg']['recall']:.4f}")
    print(f"  F1 Score: {judgment_metrics['macro_avg']['f1']:.4f}")

    print(f"\nWeighted Average:")
    print(f"  Precision: {judgment_metrics['weighted_avg']['precision']:.4f}")
    print(f"  Recall: {judgment_metrics['weighted_avg']['recall']:.4f}")
    print(f"  F1 Score: {judgment_metrics['weighted_avg']['f1']:.4f}")

    print("\nJudgment false negative docket IDs:")
    for docket_id in judgment_metrics['false_negatives']:
        print(docket_id)
        
    # print(f"\nMetrics for Patronus:")
    # print(f"Precision: {patronus_metrics['precision']}")
    # print(f"Recall: {patronus_metrics['recall']}")
    # print(f"F1 Score: {patronus_metrics['f1']}")
    # print("\nPatronus false negative docket IDs:")
    # for docket_id in patronus_metrics['false_negatives']:
    #     print(docket_id)
