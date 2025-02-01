from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ScoreType(Enum):
    CONTRADICTION = "Contradiction"
    INFO_COVERAGE = "Info Coverage"


class ContradictionVerdict(BaseModel):
    # yes, no, or idk
    verdict: str
    reason: Optional[str] = Field(default=None)


class InfoCoverageVerdict(BaseModel):
    summary_verdict: str
    original_verdict: str
    question: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContradictionVerdict]


class Questions(BaseModel):
    questions: List[str]


class Answers(BaseModel):
    answers: List[str]


class Reason(BaseModel):
    reason: str


class SummarizationTemplate:
    @staticmethod
    def generate_reason(contradictions, redundancies, questions, score):
        return f"""==== TASK INSTRUCTIONS ====
An LLM has been tasked to summarize a text. You will be provided with the following: 
1) information in the LLM's summary contradicting the original text 
2) extra information in the LLM's summary not mentioned in the original text
3) [Optional] questions that cannot be answered by the LLM's summary
4) the summarization score, which is a 0-1 score indicating how good the summary is to the original text (higher the better)

YOUR TASK is to use this info to explain how well the LLM performed at summarizing the text. 
Please CLEARLY and CONCISELY justify the score based on the provided information.  

==== FORMATTING YOUR ANSWER ====
Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <summarization_score> because <your_reason>."
}}

For 'None' values in contradictions, extra information, or questions that the original text can answer but not the summary, DON'T mention anything and instead offer some praise.

==== EXAMPLES ====
---- START OF EXAMPLE 1 ----
Example Contradictions:
["The text claims Marie Curie won the Nobel Prize in Chemistry in 1903, but she actually won it in Physics that year", "The summary states she worked alone, but the original text mentions she collaborated with her husband Pierre"]

Example Extra Information:
["The summary mentions she taught at Oxford University, but this is not mentioned in the original text"]

Example Questions Original Text Can Answer But Summary Cannot:
["What other awards did Marie Curie receive besides the Nobel Prize?"]

Example Score: 0.65

Example Response:
{{
    "reason": "The score of 0.65 reflects issues with factual accuracy and coverage. The summary contains two factual errors about Curie's Nobel Prize field and her collaboration status, while also making unverified claims about Oxford. The summary also fails to address key questions about her other awards."
}}
---- END OF EXAMPLE 1 ----
---- START OF EXAMPLE 2 ----
Example Contradictions:
["The summary states Shakespeare wrote 40 plays, but the original text clearly states he wrote 37 plays"]

Example Extra Information:
["The summary claims Shakespeare attended Oxford University, but this is not mentioned anywhere in the original text"]

Example Questions Original Text Can Answer But Summary Cannot:
None

Example Score: 0.82

Example Response:
{{
    "reason": "The score of 0.82 reflects a generally good summary with a few issues. While the summary contains one factual error about the number of Shakespeare's plays and makes an unverified claim about Oxford attendance, it successfully covers the key information from the original text without missing any important details."
}}
---- END OF EXAMPLE 2 ----

==== YOUR TURN ====
Summarization Score:
{score}

Contradicting Information in the original text:
{contradictions}

Extra Information not mentioned in the original text:
{redundancies}
"""

    @staticmethod
    def generate_answers(questions, text):
        return f"""==== TASK INSTRUCTIONS ====
You will be provided with a passage of text and an accompanying list of questions. 
Your task is to determine whether the provided text contains sufficient information to answer each question by choosing 'yes' or 'no' for each question. 
To clarify, you should choose 'yes' if the provided text contains sufficient information to answer the question, and 'no' otherwise.

==== FORMATTING YOUR ANSWER ====
You should generate a JSON with key 'answers', which is a list of strings that determines whether the provided text contains sufficient information to answer EACH question. 
Since you are determining a verdict for each question, the length of 'answers' SHOULD BE STRICTLY EQUAL to that of the questions.

==== EXAMPLES ====
---- START OF EXAMPLE 1 ----
Example Text: The Eiffel Tower was completed in 1889 for the World's Fair in Paris. It stands 324 meters tall and was named after engineer Gustave Eiffel.
Example Questions: ["Does the text contain information about when the Eiffel Tower was built?"]
Example Answers:
{{
    "answers": ["yes"]
}}
---- END OF EXAMPLE 1 ----
---- START OF EXAMPLE 2 ----
Example Text: "The Statue of Liberty was a gift from France to the United States. It was dedicated in 1886 and stands on Liberty Island in New York Harbor."
Example Questions: ["Does the text mention who gave the Statue of Liberty?", "Does the text indicate where the statue is located?"]
Example Answers:
{{
    "answers": ["yes", "yes"]
}}
---- END OF EXAMPLE 2 ----
===== END OF EXAMPLES ======

==== YOUR TURN ====
Text:
{text}

Questions:
{questions}

JSON:
"""

    @staticmethod
    def generate_questions(text, n):
        return f"""==== TASK INSTRUCTIONS ====
Based on the given text, generate {n} closed-ended questions that can be answered with either a 'yes' or 'no'. 
The questions generated should ALWAYS result in a 'yes' based on the given text. 
        
==== FORMATTING YOUR ANSWER ====
Only return a JSON with a 'questions' key, which is a list of strings. The questions need to be closed ended, meaning they are answered with either 'yes' or 'no'. 
Remember that for this task, we should be able to use the given text to answer 'yes' for each question you generate.

==== EXAMPLES ====
---- START OF EXAMPLE 1 ----
Example Text: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
N = 2 questions

Example Answers:
{{
    "questions": ["Is there enough information about Einstein's nationality?", "Is there enough information to know Einstein's Nobel Prize year?"]
}}
---- END OF EXAMPLE 1 ----
---- START OF EXAMPLE 2 ----
Example Text: "The Great Wall of China was built over many centuries by different Chinese dynasties. Construction began more than 2,000 years ago and continued through multiple dynasties. The wall stretches for thousands of miles across China's northern borders."
N = 2 questions

Example Answers:
{{
    "questions": ["Does the text provide information about when construction of the Great Wall began?", "Is there information about the Great Wall's location relative to China?"]
}}
---- END OF EXAMPLE 2 ----
===== END OF EXAMPLES ======

==== YOUR TURN ====
Text:
{text}

N = {n}

JSON:
"""

    @staticmethod
    def generate_contradiction_verdicts(original_text, summary_claims):
        return f"""==== TASK INSTRUCTIONS ====

You will be provided with a text and a list of summary claims. The list of claims is drawn from a summary of the original text. 
Your task is to determine whether each claim is factually consistent with the original text.

NOTE: You should NOT use your prior knowledge in your judgment. It does NOT matter if the claim is correct; we're just interested in whether the claim is factually consistent with the original text. 
Claims that is not backed up due to a lack of information/is not mentioned in the summary MUST be answered 'idk'. 
Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.

==== FORMATTING YOUR ANSWER ====
You should format your answer JSON with a key 'verdicts', which is a list of JSON objects. Each JSON object corresponds to a claim in the summary claims, and should have 2 fields: 'verdict' and 'reason'. 
The 'verdict' key should be EXACTLY one of 'yes', 'no', or 'idk', which represents whether the given summary claim agrees with the original text. 
The 'reason' key should be a string that provides a justification for the verdict. You should reference the original text in your reason where appropriate.

Since you are determining a verdict for each claim, the length of 'verdicts' SHOULD BE EXACTLY EQUAL to that of the summary claims.


==== EXAMPLE ====
Example Original Text: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
Example Summary Claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

Example:
{{
    "verdicts": [
        {{
            "verdict": "idk",
            "reason": "The original text does not mention Barack Obama at all, let alone his racial features."
        }},
        {{
            "verdict": "idk",
            "reason": "The original text does not mention Zurich, nor does it mention Zurich being in London"
        }},
        {{
            "verdict": "yes",
            "reason": "The original text directly states that Einstein won the Nobel Prize for his discovery of the photoelectric effect, which matches this claim."
        }},
        {{
            "verdict": "no",
            "reason": "The summary claims Einstein won the Nobel Prize in 1969, which is untrue as the original text states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The summary claims Einstein is a Germen chef, which is not correct as the original text states he was a German scientist instead."
        }}
    ]  
}}
===== END OF EXAMPLE ======


==== YOUR TURN ====
Original Text:
{original_text}

Summary Claims:
{summary_claims}

JSON:
"""
