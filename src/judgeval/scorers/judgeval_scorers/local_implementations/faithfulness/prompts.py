from typing import List, Optional
from pydantic import BaseModel, Field


class FaithfulnessVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[FaithfulnessVerdict]


class Truths(BaseModel):
    truths: List[str]


class Claims(BaseModel):
    claims: List[str]


class Reason(BaseModel):
    reason: str


class FaithfulnessTemplate:
    @staticmethod
    def find_claims(text):
        return f"""==== TASK INSTRUCTIONS ====
You will be provided with a passage of text. Based on the text, your task is to generate a comprehensive list of ALL CLAIMS that can be inferred from the text.  
For every claim that you derive from the text, provide the source of the claim via quoting the original text. Please try to extract EVERY CLAIM that is in the original text; priortize generating the most claims rather than being concise. 
You should NOT include any prior knowledge, and take the text at face value when extracting claims.

==== FORMATTING YOUR ANSWER ====
Please return your answer in JSON format, with the "claims" key as a list of JSON objects with keys "claim" and "quote". No words or explanation beyond the output JSON is needed.

==== EXAMPLES ====

---- START OF EXAMPLE 1 ----
Example Text: 
"Einstein won the nobel prize in 1968 for his discovery of the photoelectric effect."

Example JSON: 
{{
    "claims": [
        {{
             "claim": "Einstein won the nobel prize for his discovery of the photoelectric effect.",
             "quote": "Einstein won the nobel prize in 1968 for his discovery of the photoelectric effect."
        }},
        {{
             "claim": "Einstein won the nobel prize in 1968.",
             "quote": "Einstein won the nobel prize in 1968 for his discovery of the photoelectric effect."
        }}
    ]  
}}
---- END OF EXAMPLE 1 ----

---- START OF EXAMPLE 2 ----
Example Text: "The Wright brothers successfully flew the first powered airplane on December 17, 1903, in Kitty Hawk, North Carolina."

{{
    "claims": [
        {{
            "claim": "The Wright brothers flew the first powered airplane.",
            "quote": "The Wright brothers successfully flew the first powered airplane on December 17, 1903, in Kitty Hawk, North Carolina."
        }},
        {{
            "claim": "The Wright brothers made their flight in Kitty Hawk, North Carolina.",
            "quote": "The Wright brothers successfully flew the first powered airplane on December 17, 1903, in Kitty Hawk, North Carolina."
        }},
        {{
            "claim": "The first powered airplane flight occurred on December 17, 1903.",
            "quote": "The Wright brothers successfully flew the first powered airplane on December 17, 1903, in Kitty Hawk, North Carolina."
        }}
    ]
}}
---- END OF EXAMPLE 2 ----

---- START OF EXAMPLE 3 ----
Example Text:
"The Great Wall of China was built over many centuries by different Chinese dynasties. Construction began more than 2,000 years ago during the Warring States period. The most famous sections were built during the Ming Dynasty. The wall stretches for thousands of miles across northern China and was primarily built for military defense."

Example JSON:
{{
    "claims": [
        {{
            "claim": "The Great Wall of China was built by multiple Chinese dynasties",
            "quote": "The Great Wall of China was built over many centuries by different Chinese dynasties."
        }},
        {{
            "claim": "Construction of the Great Wall began over 2,000 years ago",
            "quote": "Construction began more than 2,000 years ago during the Warring States period."
        }},
        {{
            "claim": "Construction started during the Warring States period",
            "quote": "Construction began more than 2,000 years ago during the Warring States period."
        }},
        {{
            "claim": "The most well-known parts of the wall were constructed during the Ming Dynasty",
            "quote": "The most famous sections were built during the Ming Dynasty."
        }},
        {{
            "claim": "The Great Wall extends for thousands of miles",
            "quote": "The wall stretches for thousands of miles across northern China"
        }},
        {{
            "claim": "The wall is located in northern China",
            "quote": "The wall stretches for thousands of miles across northern China"
        }},
        {{
            "claim": "The Great Wall was constructed for defensive military purposes",
            "quote": "was primarily built for military defense."
        }}
    ]
}}
---- END OF EXAMPLE 3 ----

==== END OF EXAMPLES ====

==== YOUR TURN ====

Example Text:
{text}

JSON:
"""

    @staticmethod
    def create_verdicts(claims, retrieval_context):
        return f"""==== TASK INSTRUCTIONS ====
You will be provided with a list of claims from an LLM's output text, accompanied by the retrieval documents that the LLM used to generate the output. 
I'm pretty sure that many of the claims are factually contradictory to the retrieval context, but I want you to double check that I'm right. 
For each claim, choose one of ("yes", "no", or "idk") to represent whether the claim is correct based on the retrieval context. 
YOU SHOULD be very scrutinous--if any part of the claim is contradicted by the retrieval context, you should choose "no". Think really hard about finding the contradictions, since they can be subtle!

Choose 'no' if the retrieval context CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGMENT.
Claims made using vague, suggestive, or speculative language such as 'may have', 'possibility due to', do NOT count as a contradiction.
Claims that are fuzzy based on lack of information MUST BE ANSWERED with 'idk'.

==== FORMATTING YOUR ANSWER ====
Please return your answer in JSON format, with the 'verdicts' key as a list of JSON objects. Each JSON object should have 2 fields: 'verdict' and 'reason'. 
The 'verdict' key should be either 'yes', 'no', or 'idk', which states WHETHER THE CLAIM AGREES with the context. 
The 'reason' key should be a string explaining why the claim is 'yes', 'no', or 'idk'. Make specific reference to the retrieval context to justify your verdict.

==== EXAMPLES ====
---- START OF EXAMPLE 1 ----
Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
Example claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "idk",
            "reason": "The claim about Barack Obama's ethnicity cannot be verified from the given retrieval context as it contains no information about Barack Obama."
        }},
        {{
            "verdict": "idk", 
            "reason": "The claim about Zurich being a city in London cannot be verified from the given retrieval context as it contains no information about Zurich or London."
        }},
        {{
            "verdict": "yes",
            "reason": "The retrieval context confirms that Einstein won the Nobel Prize for discovering the photoelectric effect."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the retrieval context states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein is a Germen chef, which is not correct as the retrieval context states he was a German scientist instead."
        }}
    ]  
}}
---- END OF EXAMPLE 1 ----
---- START OF EXAMPLE 2 ----
Example retrieval contexts: "The Great Wall of China was built over many centuries by different Chinese dynasties. Construction began more than 2,000 years ago. The wall stretches for thousands of miles across China's northern borders. Most of the existing wall was built during the Ming Dynasty."
Example claims: ["The Great Wall was built in a single year.", "The Great Wall may have taken centuries to complete.", "The Great Wall was built by the Romans.", "The Great Wall is located in China's northern region.", "The Great Wall is 100 meters long."]

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "no",
            "reason": "The claim that the Great Wall was built in a single year directly contradicts the retrieval context, which states it was built over many centuries."
        }},
        {{
            "verdict": "yes",
            "reason": "The retrieval context confirms that the Great Wall was built over many centuries by different Chinese dynasties."
        }},
        {{
            "verdict": "no",
            "reason": "The claim states the Romans built the wall, which contradicts the retrieval context that specifies it was built by Chinese dynasties."
        }},
        {{
            "verdict": "yes",
            "reason": "The retrieval context explicitly states that the wall stretches across China's northern borders."
        }},
        {{
            "verdict": "no",
            "reason": "The claim that the wall is 100 meters long contradicts the retrieval context which states it stretches for thousands of miles."
        }}
    ]
}}
---- END OF EXAMPLE 2 ----
==== END OF EXAMPLES ====

==== YOUR TURN ====
Retrieval Contexts:
{retrieval_context}

Claims:
{claims}

JSON:
"""

    @staticmethod
    def justify_reason(score, contradictions):
        return f"""==== TASK INSTRUCTIONS ====
You will be provided with a list of contradictions and a faithfulness score. 
The list of contradictions will be references to statements made by a RAG generator that contradicted one or more document(s) from the retrieval context. 
- To clarify, the LLM produced an `actual output` that contained claims that contradicted the `retrieval context` used to produce the output.
The faithfulness score is a 0 - 1 float (1 is highest) indicating how factually consistent the RAG generator's output is to the retrieval context.

Your task is to CLEARLY and CONCISELY summarize the contradictions to justify the score. 
If there are no contradictions, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying). 
Your reason MUST use information from the contradictions in your reason.


==== FORMATTING YOUR ANSWER ====
Please make sure to only return in JSON format, with the 'reason' key providing the reason.

Example JSON:
{{
    "reason": "The score is <faithfulness_score> because <your_reason>."
}}

==== EXAMPLE ====
Example Contradictions:
[
    {{
        "verdict": "no", 
        "reason": "The output claims Marie Curie was born in Russia, but the context clearly states she was born in Warsaw, Poland."
    }},
    {{
        "verdict": "no",
        "reason": "The output states Marie Curie discovered uranium, but the context indicates she discovered radium and polonium."
    }}
]

Example Faithfulness Score:
0.60

Example Response:
{{
    "reason": "The score is 0.60 because the output made two significant factual errors: incorrectly stating Marie Curie was born in Russia instead of Poland, and wrongly attributing the discovery of uranium to her instead of radium and polonium."
}}

==== YOUR TURN ====
Faithfulness Score:
{score}

Contradictions:
{contradictions}

JSON:
"""
