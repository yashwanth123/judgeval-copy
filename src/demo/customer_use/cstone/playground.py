import csv
import os
import openai
import json
from openai import OpenAI
import anthropic
import time
from together import Together
import concurrent.futures

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
                    
                    Provide your response in JSON format (just the brackets no other text) with only one field contradiction (yes or no)
                    """
                    # Query the OpenAI API  

                    # Function to call OpenAI API
                    def call_openai(model, prompt):
                        client = openai.Client()
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        return response.choices[0].message.content

                    # Function to call Anthropic API
                    def call_anthropic(prompt):
                        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                        response = client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=1024,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        return response.content[0].text

                    # Prepare to collect responses
                    list_of_4o_responses = []
                    list_of_sonnet_responses = []
                    list_of_o1_responses = []

                    # Use ThreadPoolExecutor to perform API calls in parallel
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Prepare futures for OpenAI API calls
                        futures_4o = [executor.submit(call_openai, "gpt-4o", prompt) for _ in range(2)]
                        
                        # Prepare futures for Anthropic API calls
                        futures_sonnet = [executor.submit(call_anthropic, prompt) for _ in range(2)]
                        
                        # Prepare futures for O1 API calls
                        futures_o1 = [executor.submit(call_openai, "o1-mini", prompt) for _ in range(2)]

                        # Collect OpenAI responses
                        for future in concurrent.futures.as_completed(futures_4o):
                            list_of_4o_responses.append(future.result())

                        # Collect Anthropic responses
                        for future in concurrent.futures.as_completed(futures_sonnet):
                            list_of_sonnet_responses.append(future.result())

                        # Collect O1 responses
                        for future in concurrent.futures.as_completed(futures_o1):
                            list_of_o1_responses.append(future.result())
                    
                    print(list_of_4o_responses)
                    print(list_of_o1_responses)
                    print(list_of_sonnet_responses)

                    total_responses = list_of_4o_responses + list_of_o1_responses + list_of_sonnet_responses

                    prompt_o1_preview = f"""
                    Given the following JSONs, if at least 2 of the JSONs say there is a contradiction, return "yes", otherwise return "no". 
                    {total_responses}

                    Provide your response in JSON format (just the brackets no other text) with fields contradiction (yes or no). Make sure to only have 1 field.
                    """
                    client = openai.Client()
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

        
