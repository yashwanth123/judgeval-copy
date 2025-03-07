import csv
from openai import OpenAI
from typing import List

def extract_data(file_path, 
                               docket_id_col='docket_id', 
                               excerpts_col='excerpts', 
                               llm_response_col='LLM_raw_response', 
                               label_col='correct', 
                               note_col='RZ_note',
                               filter_type='incorrect_only'):
    """
    Extracts data from a CSV file and filters rows based on correctness.
    
    Args:
        file_path (str): Path to the CSV file
        docket_id_col (str): Column name for docket ID
        excerpts_col (str): Column name for excerpts
        llm_response_col (str): Column name for LLM response
        label_col (str): Column name for correctness indicator
        note_col (str): Column name for notes
        filter_type (str): Type of filtering to apply. Options:
                          'incorrect_only': Only return incorrect judgments
                          'correct_only': Only return correct judgments
                          'both': Return both correct and incorrect judgments
    
    Returns:
        list: List of dictionaries containing the extracted data

        {
        "docket_id": "1234567890",
        "excerpts": "excerpts",
        "LLM_raw_response": "LLM_raw_response",
        "correct": "correct",
        "RZ_note": "RZ_note"
        }
    """
    extracted_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            print(f"CSV Headers: {reader.fieldnames}")
            
            for row in reader:
                # Extract the required columns
                data = {
                    'docket_id': row.get(docket_id_col, ''),
                    'excerpts': row.get(excerpts_col, ''),
                    'LLM_raw_response': row.get(llm_response_col, ''),
                    'correct': row.get(label_col, ''),
                    'RZ_note': row.get(note_col, '')
                }
                
                # Filter based on the filter_type
                if filter_type == 'incorrect_only' and data['correct'].upper() == 'FALSE':
                    extracted_data.append(data)
                elif filter_type == 'correct_only' and data['correct'].upper() == 'TRUE':
                    extracted_data.append(data)
                elif filter_type == 'both':
                    extracted_data.append(data)
        
        return extracted_data
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return []


def find_failure_reason(task_instruction: str, llm_response: str, excerpts: str, client=None) -> str:
    """
    Reasons over why the LLM made the wrong judgment.
    
    Args:
        task_instruction: The instruction given for the task
        llm_response: The LLM's response that was incorrect
        excerpts: The context provided to the LLM
        client: Optional OpenAI client instance. If None, a new client will be created.
    
    Returns:
        Analysis of why the LLM made the wrong judgment
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()
        
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are an analytical lawyer breaking down why an LLM has made a mistake for a task. The task is: {task_instruction}. \n **YOUR JOB** is to analyze the LLM's response to this task and the excerpts it had access to in order to solve the task. The LLM definitely got the task wrong, whether it be hallucinating information, misinterpreting the excepts, or not following the actual task. YOU NEED TO FIND WHY."},
            {"role": "user", "content": f"LLM Response (INCORRECT): {llm_response}\n\nExcerpts the LLM had to work with: {excerpts}"}
        ],
    )
    return response.choices[0].message.content


def find_failure_reasons_parallel(task_instruction: str, data_list: list, response_key: str = 'LLM_raw_response', 
                                 excerpts_key: str = 'excerpts', max_workers: int = 10) -> list:
    """
    Parallelizes the process of finding failure reasons for multiple examples.
    
    Args:
        task_instruction: The instruction given for the task
        data_list: List of dictionaries containing LLM responses and excerpts
        response_key: Key in the dictionaries for LLM responses
        excerpts_key: Key in the dictionaries for excerpts
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of failure reason analyses
    """
    import concurrent.futures
    from openai import OpenAI
    
    client = OpenAI()
    results = []
    
    def process_item(item):
        return find_failure_reason(
            task_instruction=task_instruction,
            llm_response=item.get(response_key, ''),
            excerpts=item.get(excerpts_key, ''),
            client=client
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data_list}
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    # Write all the results to a file, with newlines in between
    with open("failure_analysis_results.txt", "w") as f:
        for i, result in enumerate(results):
            f.write(f"Failure Reason for Example {i+1}:\n")
            f.write(result)
            f.write("\n\n" + "-"*80 + "\n\n")
    
    return results
    

def inference(excerpts: str, llm_response: str, task_instruction: str, model: str = "gpt-4o", client=None) -> str:
    """
    {
        "docket_id": "1234567890",
        "excerpts": "excerpts",
        "LLM_raw_response": "LLM_raw_response",
        "correct": "correct",
        "RZ_note": "RZ_note"
        }
    """

    # prompt = f"""You are an analytical lawyer breaking down whether or not an LLM has made a hallucination in its response to a task. 
    # You will be provided with a set of instructions for the task, a set of excerpts the LLM used to complete the task, and the LLM's response to the task. 
    # You must analyze whether based on the information provided, the LLM has made a hallucination. A hallucination is defined as one or more of the following mistakes:
    # - The LLM's response contradicts the information provided in the excerpts
    # - The LLM's response misinterprets the information provided in the excerpts
    # - The LLM's response diverges from the task instructions, such as answering a different question than the one provided in the instructions or addressing a different task than the one provided.

    # ==== FORMATTING INSTRUCTIONS ====
    # First, analyze any possible contradictions and misinterpretations of the information provided in the excerpts.
    # Next, analyze whether the LLM's response diverges from the task instructions.
    # Finally, based on the above analysis, determine whether the LLM has made a hallucination.

    # **End your response with <answer>True</answer> if the LLM has made a hallucination, and <answer>False</answer> if it has not.**
    # """

    prompt = f"""You are an analytical lawyer breaking down whether or not an LLM has made a hallucination in its response to a task. I'M PRETTY SURE THERE'S A HALLUCINATION, BUT I WANT YOU TO DOUBLE CHECK.
    You will be provided with a set of instructions for the task, a set of excerpts the LLM used to complete the task, and the LLM's response to the task. 
    You must analyze whether based on the information provided, the LLM has made a hallucination. A hallucination is defined as one or more of the following mistakes:
    - The LLM's response contradicts the information provided in the excerpts
    - The LLM's response misinterprets the information provided in the excerpts
    - The LLM's response diverges from the task instructions, such as answering a different question than the one provided in the instructions or addressing a different task than the one provided.

    Again, I'm pretty sure there's a hallucination in the model response, but I COULD BE WRONG!
    ==== FORMATTING INSTRUCTIONS ====
    First, analyze any possible contradictions and misinterpretations of the information provided in the excerpts.
    Next, analyze whether the LLM's response diverges from the task instructions.
    Finally, based on the above analysis, determine whether the LLM has made a hallucination.

    **End your response with <answer>True</answer> if the LLM has made a hallucination, and <answer>False</answer> if it has not made a hallucination.**
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"""==== YOUR TURN ====
    TASK INSTRUCTIONS: {task_instruction}

    ----------------
    EXCERPTS: {excerpts}

    ----------------
    LLM RESPONSE: {llm_response}"""}
        ],
    ).choices[0].message.content

    return response

def inference_parallel(data_list: list, task_instruction: str, model: str = "gpt-4o", 
                      excerpts_key: str = 'excerpts', response_key: str = 'LLM_raw_response', 
                      max_workers: int = 10, log_file: str = "inference_results.txt") -> list:
    """
    Parallelizes the process of running inference on multiple examples.
    
    Args:
        data_list: List of dictionaries containing LLM responses and excerpts
        task_instruction: The instruction given for the task
        model: The OpenAI model to use
        excerpts_key: Key in the dictionaries for excerpts
        response_key: Key in the dictionaries for LLM responses
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of inference results
    """
    import concurrent.futures
    from openai import OpenAI
    
    client = OpenAI()
    results = []
    
    def process_item(item):
        return inference(
            excerpts=item.get(excerpts_key, ''),
            llm_response=item.get(response_key, ''),
            task_instruction=task_instruction,
            model=model,
            client=client
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data_list}
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
    
    with open(log_file, "w") as f:
        for i, result in enumerate(results):
            f.write(f"Inference Result for Example {i+1}:\n")
            f.write(result)
            f.write("\n\n" + "-"*80 + "\n\n")
    
    return results


def extract_answer(text: str) -> bool:
    """
    Extracts the answer value between <answer> and </answer> tags from a string.
    
    Args:
        text: String containing an answer enclosed in <answer> tags
        
    Returns:
        bool: True if the answer is "True", False if the answer is "False"
        
    Raises:
        ValueError: If the answer is not found or is not "True" or "False"
    """
    import re
    
    # Find the content between <answer> and </answer> tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    if not match:
        raise ValueError("No answer found between <answer> and </answer> tags")
    
    answer_text = match.group(1).strip()
    
    # Check if the answer is valid
    if answer_text.lower() == "true":
        return True
    elif answer_text.lower() == "false":
        return False
    else:
        raise ValueError(f"Invalid answer: '{answer_text}'. Answer must be 'True' or 'False'")



if __name__ == "__main__":
    ### CONFIG
    NUM_TRIALS = 3
    MODEL_NAME = "gpt-4o"
    FILTER_TYPE = "correct_only"

    print(f"Running inference for {NUM_TRIALS} trials with model {MODEL_NAME} and filter type {FILTER_TYPE}")

    file_path = "/Users/alexshan/Desktop/judgment_labs/judgeval/src/demo/customer_use/cstone/JudgmentDemo/wh-driver-amend-charter.csv"
    task_instruction_file = "/Users/alexshan/Desktop/judgment_labs/judgeval/src/demo/customer_use/cstone/JudgmentDemo/prompts/driver_amend.txt"
    with open(task_instruction_file, 'r') as file:
        task_instruction = file.read()


    incorrect_row_data: List[dict] = extract_data(
        file_path,
        excerpts_col="content",
        llm_response_col="raw_response",
        label_col="correct",
        note_col="RZ_note",
        filter_type=FILTER_TYPE
    )
    print(f"Number of fetched incorrect judgments: {len(incorrect_row_data)}")
    

    for _ in range(NUM_TRIALS):
        inference_results: List[str] = inference_parallel(
            data_list=incorrect_row_data,
            task_instruction=task_instruction,
            model=MODEL_NAME,
            excerpts_key="excerpts",
            response_key="LLM_raw_response",
            max_workers=55
        )

        hallucination_results: List[bool] = [extract_answer(result) for result in inference_results]

        # Count the number of True and False results
        true_count = hallucination_results.count(True)
        false_count = hallucination_results.count(False)
        
        # Print the counts
        print(f"Hallucination Results Summary:")
        print(f"  True: {true_count} ({true_count/len(hallucination_results)*100:.2f}%)")
        print(f"  False: {false_count} ({false_count/len(hallucination_results)*100:.2f}%)")
        print(f"  Total: {len(hallucination_results)}")

