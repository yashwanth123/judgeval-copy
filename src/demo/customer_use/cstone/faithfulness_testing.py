import os
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

from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from patronus import Client as PatronusClient
# from autoevals import Faithfulness
# from braintrust import Eval
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#JUDGMENT_API_KEY = os.getenv("JUDGMENT_API_KEY")  

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

with open(os.path.join(os.path.dirname(__file__), "cstone_data.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row
    data = list(reader)

# Test the model
client = JudgmentClient()
examples = []
scorer = FaithfulnessScorer(threshold=0.7)
ragas_scores = []
patronus_client = PatronusClient(api_key=os.getenv("PATRONUS_API_KEY"))
patronus_results = []

patronus_faith = []
ragas_faith = []
judgment_faith = []

for row in data:
    docket_id, excerpts, raw_response, quote, is_class_action, note, is_hallucination = row

    result = patronus_client.evaluate(
        evaluator="lynx-small",
        criteria="patronus:hallucination",
        evaluated_model_input="Does the context identify a class action lawsuit?",
        evaluated_model_output=raw_response,
        evaluated_model_retrieved_context=[excerpts],
        tags={"scenario": "cstone"},
    )
    patronus_faith.append(result.score_raw)
    patronus_results.append(result)
    ragas_score = SingleTurnSample(
        user_input=str(docket_id),
        response=raw_response,
        retrieved_contexts=[excerpts],
    )
    ragas_scores.append(ragas_score)
    example = Example(
        input=str(docket_id),
        actual_output=raw_response,
        retrieval_context=[excerpts],
    )
    examples.append(example)

#print(patronus_results)

async def run_ragas_evaluation(dataset):
    ragas_scorer = Faithfulness(llm=evaluator_llm)
    for score in dataset:
        res = await ragas_scorer.single_turn_ascore(score)
        #print(f"Faithfulness Score: {res}")
        ragas_faith.append(res)
    

if __name__ == "__main__":
    dataset = EvaluationDataset(ragas_scores)
    
    # Run async evaluation
    ragas_results = asyncio.run(run_ragas_evaluation(dataset))
    
    output = client.run_evaluation(
        model="gpt-4o",
        examples=examples,
        scorers=[scorer],
        eval_run_name="cstone-basic-test",
        project_name="cstone_faithfulness_testing",
        override=True,
        use_judgment=False,
    )
    
    for result in output:
        score = result.scorers_data[0].score  # Get the faithfulness score
        judgment_faith.append(score)
    
    # print(f"Faithfulness scores: {judgment_faith}")

    # print(patronus_faith)
    # print(ragas_faith)
    # print(judgment_faith)

    # Convert scores to boolean based on 0.7 threshold
    bool_patronus = [score < 0.7 for score in patronus_faith]
    bool_ragas = [score < 0.7 for score in ragas_faith]
    bool_judgment = [score < 1 for score in judgment_faith]

    # Read the existing CSV first to get hallucination data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "cstone_data.csv"))
    is_hallucination = df['is_hallucination'].tolist()  # Convert column to list
    
    # Calculate TP and FN using list comprehension
    TP_patronus = sum(1 for h, p in zip(is_hallucination, bool_patronus) if h and p)
    FN_patronus = sum(1 for h, p in zip(is_hallucination, bool_patronus) if h and not p)
    
    TP_ragas = sum(1 for h, r in zip(is_hallucination, bool_ragas) if h and r)
    FN_ragas = sum(1 for h, r in zip(is_hallucination, bool_ragas) if h and not r)
    
    TP_judgment = sum(1 for h, j in zip(is_hallucination, bool_judgment) if h and j)
    FN_judgment = sum(1 for h, j in zip(is_hallucination, bool_judgment) if h and not j)

    recall_patronus = TP_patronus / (TP_patronus + FN_patronus)
    recall_ragas = TP_ragas / (TP_ragas + FN_ragas)
    recall_judgment = TP_judgment / (TP_judgment + FN_judgment)

    print(f"Recall scores:")
    print(f"Patronus: {recall_patronus}")
    print(f"Ragas: {recall_ragas}")
    print(f"Judgment: {recall_judgment}")

    