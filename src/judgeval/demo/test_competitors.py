from dotenv import load_dotenv
from patronus import Client
import os 
import asyncio
import time
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()

PATRONUS_API_KEY = os.getenv("PATRONUS_API_KEY")

client = Client(api_key=PATRONUS_API_KEY)

# Initialize clients
openai_client = OpenAI()
anthropic_client = Anthropic()

async def make_upper(input: str) -> str:
    output = input.upper()
    result = client.evaluate(
        evaluator="answer-relevance",
        criteria="patronus:answer-relevance",
        evaluated_model_input=input,
        evaluated_model_output=output,
        threshold=0.5,
        model="gpt-4o-mini",
        log_results=True
    )
    return output

def llm_call(input):
    time.sleep(1.3)
    return "We have a 30 day full refund policy on shoes."

async def answer_user_question(input):
    output = llm_call(input)
    result = client.evaluate(
        evaluator="answer-relevance",
        criteria="patronus:answer-relevance",
        evaluated_model_input=input,
        evaluated_model_output=output,
        evaluated_model_retrieved_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        expected_output="We offer a 30-day full refund at no extra cost.",
        threshold=0.5,
        model="gpt-4o-mini",
        log_results=True
    )
    return output

async def make_poem(input: str) -> str:
    try:
        # Using Anthropic API
        anthropic_response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": input}],
            max_tokens=30
        )
        anthropic_result = anthropic_response.content[0].text
        
        result = client.evaluate(
            evaluator="answer-relevance",
            criteria="patronus:answer-relevance",
            evaluated_model_input=input,
            evaluated_model_output=anthropic_result,
            threshold=0.5,
            model="gpt-4o-mini",
            log_results=True
        )
        
        # Using OpenAI API
        openai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Make a short sentence with the input."},
                {"role": "user", "content": input}
            ]
        )
        openai_result = openai_response.choices[0].message.content
        
        return f"{anthropic_result} {openai_result}".lower()
    
    except Exception as e:
        print(f"Error generating poem: {e}")
        return ""

async def test_evaluation_mixed(input):
    upper = await make_upper(input)
    result = await make_poem(upper)
    await answer_user_question("What if these shoes don't fit?")
    return result

if __name__ == "__main__":
    test_input = "Write a poem about Nissan R32 GTR"
    asyncio.run(test_evaluation_mixed(test_input))
    
