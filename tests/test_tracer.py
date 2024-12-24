
from openai import OpenAI
from judgeval.common.tracer import Tracer, wrap_openai

import time

# Initialize the tracer
judgment = Tracer(api_key="YOUR_API_KEY")
client = wrap_openai(OpenAI())

@judgment.observe
def make_upper(input):
    time.sleep(1)
    return input.upper()

@judgment.observe
def make_lower(input):
    time.sleep(1.2)
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You must convert the input to lowercase."}, {"role": "user", "content": input}])
    return response.choices[0].message.content

@judgment.observe
def make_poem(input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a poet. Make a short haiku from the input."},
            {"role": "user", "content": input}],
    )
    return make_lower(response.choices[0].message.content)


def test_evaluation_mixed(input):
    with judgment.start_trace("test_evaluation") as trace:
        result = make_poem(make_upper(input))
        
    trace.print_trace()
    trace.save_trace()
    return result

result3 = test_evaluation_mixed("hello the world is flat")