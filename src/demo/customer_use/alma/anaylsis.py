
from dotenv import load_dotenv
import os
import csv
from judgeval.data import Example
from judgeval import JudgmentClient
from judgeval.scorers import ComparisonScorer
from typing import List
import openai
import json



def load_examples():
    """Load and parse the data from CSV file"""
    with open(os.path.join(os.path.dirname(__file__), "data.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        data = list(reader)
    
    examples = []
    for row in data:
        id, draft_text, final_text = row
        example = Example(
            input=str(id),
            actual_output=draft_text,
            expected_output=final_text,
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
    scorer1 = ComparisonScorer(threshold=1.0, criteria="Use of Evidence and Details", description="Incorporation of specific achievements, positions held, and quantitative data to substantiate claims about the applicant's capabilities")
    scorer2 = ComparisonScorer(threshold=1.0, criteria="Clarity and Specificity", description="Enhancements in the communication clarity and specificity of the applicant's achievements and contributions.")
    
    output = client.run_evaluation(
        model="gpt-4o",
        examples=examples,
        scorers=[scorer1, scorer2],
        eval_run_name="alma-basic-test4", 
        project_name="alma-basic-test",
        override=True,
    )
    return output

def find_categories(examples: List[Example]):
    """
    Find the categories of the examples
    """
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

    current_categories = []

    for example in examples:
        prompt = f"""
        I will provide you with a rough draft and a final version of a draft. The final version is a lot better than the rough draft.
        I want you to find the differences between the two drafts, and then categorize these differences into criteria. Examples of creteria should be more generic, not specific to this example. (IE Use of Evidence, Professionalism and Tone, etc.)
        This is the current list of criteria: {current_categories}. If you find that the criteria is already in the list, make sure to use the previous counts and update them.

        Place emphasis on finding these unique criteria, BE CREATIVE. I also want you to keep track of the amount of differences found for each criteria.
        If you ever find that two or more criteria are similar or fit into a more general category, you can combine them into one criteria, but make sure to update the description to include both criteria and the amount of differences found. If you ever are at more than 12 criteria, it is likely that you are getting too specific and need to combine some criteria.

        I want you to first generate all this information and reasoning behind it, and then at the end return the JSON array of objects with the following format:
        [
            {{
                "criteria": "(criteria)",
                "description": "(description of the criteria, try to make this more generic, not specific to this example)",
                "amount_of_differences": "(number)",
                "differences": "(list of differences between the two drafts for this criteria)"
            }},
            {{
                "criteria": "(criteria)",
                "description": "(description of the criteria, try to make this more generic, not specific to this example)",
                "amount_of_differences": "(number)",
                "differences": "(list of differences between the two drafts for this criteria)"
            }},
            ...
        ]

        So your response should be something like:
        (all your reasoning here)
        
        JSON RESPONSE:
        [
            (json response here)
        ]

        Here are the rough draft and final version:
        Rough Draft: {example.actual_output}
        Final Version: {example.expected_output}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        response = response[response.index('json') + len('json'):].strip()
        response = response[response.index('['):response.rindex(']')+1]

        try:
            current_categories = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {response}")
            break

        print(current_categories)
        print()


    return current_categories

    

def main():
    load_dotenv()
    examples = load_examples()
    # categories = find_categories(examples)
    run_judgment_evaluation(examples)

if __name__ == "__main__":
    main()