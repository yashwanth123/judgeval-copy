from openai import OpenAI
from together import Together
from anthropic import Anthropic
from judgeval.common.tracer import Tracer, wrap
from judgeval.constants import APIScorer
import os
import time
import asyncio
# Initialize the tracer and clients
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))
openai_client = wrap(OpenAI())
anthropic_client = wrap(Anthropic())

@judgment.observe
async def make_upper(input):
    await judgment.get_current_trace().async_evaluate(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        expected_output="We offer a 30-day full refund at no extra cost.",
        expected_tools=["refund"],
        score_type=APIScorer.FAITHFULNESS,
        threshold=0.5,
        model="gpt-4o-mini",
        log_results=True
    )
    return input.upper()

@judgment.observe
async def make_lower(input):
    await judgment.get_current_trace().async_evaluate(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        context=["User Account"],
        retrieval_context=["Password reset instructions"],
        tools_called=["authentication"],
        expected_tools=["authentication"],
        additional_metadata={"difficulty": "medium"},
        score_type=APIScorer.HALLUCINATION,
        threshold=0.5,
        model="gpt-4o-mini",
        log_results=True
    )
    return input.lower()

@judgment.observe
async def make_poem(input):
    
    # Using Anthropic API
    anthropic_response = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{
            "role": "user",
            "content": input
        }],
        max_tokens=30
    )
    anthropic_result = anthropic_response.content[0].text
    
    # Using OpenAI API
    openai_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Make a short sentence with the input."},
            {"role": "user", "content": input}
        ]
    )
    openai_result = openai_response.choices[0].message.content
    print(openai_result)
    
    return await make_lower(anthropic_result +  openai_result)

async def test_evaluation_mixed(input):
    with judgment.trace("test_evaluation") as trace:
        upper = await make_upper(input)
        result = await make_poem(upper)

    trace.save()
        
    trace.print()
    
    return result

if __name__ == "__main__":
    result3 = asyncio.run(test_evaluation_mixed("hello the world is flat"))

