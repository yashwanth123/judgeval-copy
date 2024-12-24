from openai import OpenAI
from together import Together
from anthropic import Anthropic
from judgeval.common.tracer import Tracer, wrap

import time

# Initialize the tracer and clients
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))
openai_client = wrap(OpenAI())
anthropic_client = wrap(Anthropic())

@judgment.observe
def make_upper(input):
    return input.upper()

@judgment.observe
def make_lower(input):
    return input.lower()

@judgment.observe
def make_poem(input):
    
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
    
    return make_lower(anthropic_result +  openai_result)

def test_evaluation_mixed(input):
    with judgment.trace("test_evaluation") as trace:
        result = make_poem(make_upper(input))

    trace.print()
    trace.save()
    return result

result3 = test_evaluation_mixed("hello the world is flat")
