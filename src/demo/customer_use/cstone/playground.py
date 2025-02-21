import csv
import os
import openai
import json
from openai import OpenAI
import anthropic
import time
from together import Together

if __name__ == "__main__":

    PATH_TO_CSV = os.path.join(os.path.dirname(__file__), "data.csv")
    WRITE_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

    prompts = [
        "Given the following excerpts and classification, determine if there are contradictions and explain why.",
        "Given the following excerpts and classification, determine if there are contradictions and explain why. I have a slight suspicion that there is a contradiction, but I'm not sure.",
        "Given the following excerpts and classification, determine if there are contradictions and explain why. I am pretty sure there is a contradiction.",
    ]

    for p in prompts:
        with open(WRITE_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([p])
        for i in range(1):
            with open(PATH_TO_CSV, "r") as f:
                reader = csv.reader(f)
                next(reader) # Skip the header row since we know the structure
                examples = []
                for row in reader:
                    docket_id,excerpts,LLM_raw_response,LLM_quote,is_class_action,LLM_note,final_flag,RZ_note = row
                    print(f"Docket ID: {docket_id}")
                    # print(f"Excerpts: {excerpts}")
                    # print(f"Raw Response: {LLM_raw_response}")
                    # print(f"Quote: {LLM_quote}")
                    # print(f"Is Class Action: {is_class_action}")
                    # print(f"Note: {LLM_note}")
                    # print(f"Final Flag: {final_flag}")
                    # print(f"RZ Note: {RZ_note}")


                    prompt = f"""
                    {p}
                    Excerpts: {excerpts},
                    Classification: {LLM_raw_response}
                    
                    Provide your response in JSON format (just the brackets no other text) with fields contradiction (yes or no) and explanation.
                    """
                    # Query the OpenAI API  

                    list_of_4o_responses = []
                    client = openai.Client(os.getenv("OPENAI_API_KEY"))
                    for i in range(2):
                        response_4o = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        ).choices[0].message.content
                        list_of_4o_responses.append(response_4o)

                    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    list_of_sonnet_responses = []
                    for i in range(2):
                        response_anthropic = client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=1024,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        ).content[0].text
                        list_of_sonnet_responses.append(response_anthropic)

                    list_of_o1_responses = []
                    for i in range(2):
                        response_01_mini = client.chat.completions.create(
                        model="o1-mini",
                        messages=[
                                {"role": "user", "content": prompt}
                            ]
                        ).choices[0].message.content
                        list_of_o1_responses.append(response_01_mini)


                    prompt_o1_preview = f"""
                    Given the following JSON, if the majority say there is a contradiction, return "yes", otherwise return "no". But also try to factor in the explanation of each JSON when making your decision.
                    {list_of_4o_responses}
                    {list_of_o1_responses}
                    {list_of_sonnet_responses}

                    Provide your response in JSON format (just the brackets no other text) with fields contradiction (yes or no). Make sure to only have 1.
                    """
                    response = client.chat.completions.create(
                        model="o1-preview",
                        messages=[
                            {"role": "user", "content": prompt_o1_preview}
                        ]
                    ).choices[0].message.content

                    # client = Together()

                    # response = client.chat.completions.create(
                    #     model="deepseek-ai/DeepSeek-V3",  # or "deepseek-ai/DeepSeek-R1"
                    #     messages=[{"role": "user", "content": prompt}]
                    # )

                    # response = response.choices[0].message.content
                    response = response[response.find('{'):response.rfind('}')+1]
                    print(response)


                    # Parse the JSON response
                    try:
                        analysis = json.loads(response)
                        contradiction = analysis.get('contradiction', '')
                        # explanation = analysis.get('explanation', '')
                        
                        # Append to CSV
                        with open(WRITE_CSV, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([docket_id, contradiction])
                        print(f"Added analysis for docket {docket_id}")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON response: {e}")
                    except Exception as e:
                        print(f"Error writing to CSV: {e}")
                    time.sleep(3)

        
