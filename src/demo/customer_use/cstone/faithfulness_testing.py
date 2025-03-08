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

def load_data():
    """Load and parse the data from CSV file"""
    with open(os.path.join(os.path.dirname(__file__), "cstone_data.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        data = list(reader)
    
    examples = []
    for row in data:
        docket_id, excerpts, raw_response, quote, is_class_action, note, is_hallucination = row
        example = Example(
            input=str(docket_id),
            actual_output=raw_response,
            retrieval_context=[excerpts],
        )
        examples.append(example)
        
    return examples

def run_judgment_evaluation(examples: List[Example]):
    """
    Run evaluation using JudgmentClient
    
    Args:
        examples: List of Example objects
        
    Returns:
        List of boolean values indicating if the example is a false negative
    """
    client = JudgmentClient()
    scorer = FaithfulnessScorer(threshold=1.0)
    
    output = client.run_evaluation(
        model="osiris-large",
        examples=examples,
        scorers=[scorer],
        eval_run_name="cstone-basic-test-osiris-large-1", 
        project_name="cstone_faithfulness_testing",
        override=True,
    )
    
    scores = []
    for result in output:
        score = result.scorers_data[0].score
        scores.append(score)
        
    return [score < 1 for score in scores]

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

def evaluate_predictions(predictions):
    """Calculate metrics comparing predictions to gold labels"""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "cstone_data.csv"))
    gold_labels = df['is_hallucination'].tolist()

    print(f"Gold labels: {gold_labels}")
    docket_ids = df['docket_id'].tolist()
    
    # Find false negatives
    false_negatives = [docket_id for docket_id, gold, pred in zip(docket_ids, gold_labels, predictions) 
                      if gold and not pred]
    
    # Calculate confusion matrix metrics
    TP = sum(1 for g, p in zip(gold_labels, predictions) if g and p)
    FP = sum(1 for g, p in zip(gold_labels, predictions) if not g and p) 
    TN = sum(1 for g, p in zip(gold_labels, predictions) if not g and not p)
    FN = sum(1 for g, p in zip(gold_labels, predictions) if g and not p)
    
    # Calculate final metrics
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
        
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
        
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_negatives': false_negatives
    }

if __name__ == "__main__":
    # Load data
    examples = load_data()
    
    # Run evaluations
    judgment_predictions = run_judgment_evaluation(examples)
    patronus_predictions = run_patronus_evaluation(examples)

    print(f"Judgment Predictions: {judgment_predictions}")
    print(f"Patronus Predictions: {patronus_predictions}")
    
    # Calculate and print metrics for both
    judgment_metrics = evaluate_predictions(judgment_predictions)
    patronus_metrics = evaluate_predictions(patronus_predictions)
    
    print(f"\nMetrics for Judgment:")
    print(f"Precision: {judgment_metrics['precision']}")
    print(f"Recall: {judgment_metrics['recall']}")
    print(f"F1 Score: {judgment_metrics['f1']}")
    print("\nJudgment false negative docket IDs:")
    for docket_id in judgment_metrics['false_negatives']:
        print(docket_id)
        
    print(f"\nMetrics for Patronus:")
    print(f"Precision: {patronus_metrics['precision']}")
    print(f"Recall: {patronus_metrics['recall']}")
    print(f"F1 Score: {patronus_metrics['f1']}")
    print("\nPatronus false negative docket IDs:")
    for docket_id in patronus_metrics['false_negatives']:
        print(docket_id)
