# """
# Sanity checks for judgment client functionality
# """

# import os
# from judgeval.judgment_client import JudgmentClient
# from judgeval.data import Example
# from judgeval.scorers import JudgmentScorer
# from judgeval.constants import APIScorer
# from judgeval.judges import TogetherJudge
# from judgeval.playground import CustomFaithfulnessMetric
# from judgeval.data.datasets.dataset import EvalDataset
# from dotenv import load_dotenv

# load_dotenv()

# def get_client():
#     return JudgmentClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"))


# def test_dataset(client: JudgmentClient):
#     dataset: EvalDataset = client.create_dataset()
#     dataset.add_example(Example(input="input 1", actual_output="output 1"))

#     client.push_dataset(alias="test_dataset_5", dataset=dataset, overwrite=False)
    
#     # PULL
#     dataset = client.pull_dataset(alias="test_dataset_5")
#     print(dataset)
    

# def test_run_eval(client: JudgmentClient):

#     example1 = Example(
#         input="What if these shoes don't fit?",
#         actual_output="We offer a 30-day full refund at no extra cost.",
#         retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
#         trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
#     )

#     example2 = Example(
#         input="How do I reset my password?",
#         actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
#         expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
#         name="Password Reset",
#         context=["User Account"],
#         retrieval_context=["Password reset instructions"],
#         tools_called=["authentication"],
#         expected_tools=["authentication"],
#         additional_metadata={"difficulty": "medium"}
#     )

#     scorer = JudgmentScorer(threshold=0.5, score_type=APIScorer.FAITHFULNESS)
#     scorer2 = JudgmentScorer(threshold=0.5, score_type=APIScorer.HALLUCINATION)
#     c_scorer = CustomFaithfulnessMetric(threshold=0.6)

#     PROJECT_NAME = "test_project_JOSEPH"
#     EVAL_RUN_NAME = "test_eval_JOSEPH"
#     client.run_evaluation(
#         examples=[example1, example2],
#         scorers=[scorer, c_scorer],
#         model="QWEN",
#         metadata={"batch": "test"},
#         project_name=PROJECT_NAME,
#         eval_run_name=EVAL_RUN_NAME,
#         log_results=True,
#     )

#     results = client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
#     print(f"Evaluation results for {EVAL_RUN_NAME} from database:", results)


# def test_evaluate_dataset(client: JudgmentClient):

#     example1 = Example(
#         input="What if these shoes don't fit?",
#         actual_output="We offer a 30-day full refund at no extra cost.",
#         retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
#         trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
#     )

#     example2 = Example(
#         input="How do I reset my password?",
#         actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
#         expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
#         name="Password Reset",
#         context=["User Account"],
#         retrieval_context=["Password reset instructions"],
#         tools_called=["authentication"],
#         expected_tools=["authentication"],
#         additional_metadata={"difficulty": "medium"}
#     )

#     dataset = EvalDataset(examples=[example1, example2])
#     res = client.evaluate_dataset(
#         dataset=dataset,
#         scorers=[JudgmentScorer(threshold=0.5, score_type=APIScorer.FAITHFULNESS)],
#         model="QWEN",
#         metadata={"batch": "test"},
#     )

#     print(res)

# if __name__ == "__main__":
#     # Test client functionality
#     client = get_client()
#     # print("Client initialized successfully")
#     # print("*" * 40)

#     # print("Testing dataset creation, pushing, and pulling")
#     # test_dataset(client)
#     # print("Dataset creation, pushing, and pulling successful")
#     # print("*" * 40)
    
#     print("Testing evaluation run")
#     test_run_eval(client)
#     print("Evaluation run successful")
#     print("*" * 40)
    
#     # print("Testing dataset evaluation")
#     # test_evaluate_dataset(client)
#     # print("Dataset evaluation successful")
#     # print("*" * 40)
#     # print("All tests passed successfully")
