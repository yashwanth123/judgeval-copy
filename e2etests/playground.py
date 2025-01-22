import nest_asyncio
import pprint
from typing import List, Optional, Dict, Any, Union
import json
import sys 
from rich.console import Console
from contextlib import contextmanager
from pydantic import BaseModel, Field
import re

from judgeval import langfuse
from judgeval.data.example import Example
from judgeval.judges.base_judge import judgevalJudge
from judgeval.judges.together_judge import TogetherJudge
from judgeval.judges.utils import create_judge
from judgeval.scorers.custom_scorer import CustomScorer
from judgeval.scorers.score import *

"""
Testing implementation of CustomFaithfulness
"""


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
    def generate_claims(text):
        return f"""Based on the given text, please generate a comprehensive list of FACTUAL, undisputed truths, that can inferred from the provided text.

Example:
Example Text: 
"Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."

Example JSON: 
{{
    "claims": [
        "Einstein won the noble prize for his discovery of the photoelectric effect.",
        "Einstein won the noble prize in 1968."
    ]  
}}
===== END OF EXAMPLE ======

**
IMPORTANT: Please make sure to only return in JSON format, with the "claims" key as a list of strings. No words or explanation is needed.
Only include claims that are factual, and the claims you extract should include the full context it was presented in, NOT cherry picked facts.
You should NOT include any prior knowledge, and take the text at face value when extracting claims.
**

Text:
{text}

JSON:
"""

    @staticmethod
    def generate_truths(text, extraction_limit: Optional[int] = None):
        print(extraction_limit)
        if extraction_limit is None:
            limit = " FACTUAL, undisputed truths"
        elif extraction_limit == 1:
            limit = " the single most important FACTUAL, undisputed truth"
        else:
            limit = f" the {extraction_limit} most important FACTUAL, undisputed truths per document"
        return f"""Based on the given text, please generate a comprehensive list of{limit}, that can inferred from the provided text.

Example:
Example Text: 
"Einstein won the noble prize in 1968 for his discovery of the photoelectric effect."

Example JSON: 
{{
    "truths": [
        "Einstein won the noble prize for his discovery of the photoelectric effect.",
        "Einstein won the noble prize in 1968."
    ]  
}}
===== END OF EXAMPLE ======

**
IMPORTANT: Please make sure to only return in JSON format, with the "truths" key as a list of strings. No words or explanation is needed.
Only include truths that are factual.
**

Text:
{text}

JSON:
"""

    @staticmethod
    def generate_verdicts(claims, retrieval_context):
        return f"""Based on the given claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH claim contradicts any facts in the retrieval context. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', which states whether the given claim agrees with the context. 
Provide a 'reason' ONLY if the answer is 'no'. 
The provided claim is drawn from the actual output. Try to provide a correction in the reason using the facts in the retrieval context.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
Example claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

Example:
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the retrieval context states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein is a Germen chef, which is not correct as the retrieval context states he was a German scientist instead."
        }},
    ]  
}}
===== END OF EXAMPLE ======

The length of 'verdicts' SHOULD BE STRICTLY EQUAL to that of claims.
You DON'T have to provide a reason if the answer is 'yes' or 'idk'.
ONLY provide a 'no' answer if the retrieval context DIRECTLY CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.
Claims that is not backed up due to a lack of information/is not mentioned in the retrieval contexts MUST be answered 'idk', otherwise I WILL DIE.
**

Retrieval Contexts:
{retrieval_context}

Claims:
{claims}

JSON:
"""

    @staticmethod
    def generate_reason(score, contradictions):
        return f"""Below is a list of Contradictions. It is a list of strings explaining why the 'actual output' does not align with the information presented in the 'retrieval context'. Contradictions happen in the 'actual output', NOT the 'retrieval context'.
Given the faithfulness score, which is a 0-1 score indicating how faithful the `actual output` is to the retrieval context (higher the better), CONCISELY summarize the contradictions to justify the score. 

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <faithfulness_score> because <your_reason>."
}}

If there are no contradictions, just say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
Your reason MUST use information in `contradiction` in your reason.
Be sure in your reason, as if you know what the actual output is from the contradictions.
**

Faithfulness Score:
{score}

Contradictions:
{contradictions}

JSON:
"""


@contextmanager
def metric_progress_indicator(
    metric: CustomScorer,
    async_mode: Optional[bool] = None,
    _show_indicator: bool = True,
    total: int = 9999,
    transient: bool = True,
):
    console = Console(file=sys.stderr)  # Direct output to standard error
    if _show_indicator:
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            console=console,  # Use the custom console
            transient=transient,
        ) as progress:
            progress.add_task(
                description=scorer_console_msg(metric, async_mode),
                total=total,
            )
            yield
    else:
        yield


def prettify_list(lst: List[Any]):
    if len(lst) == 0:
        return "[]"

    formatted_elements = []
    for item in lst:
        if isinstance(item, str):
            formatted_elements.append(f'"{item}"')
        elif isinstance(item, BaseModel):
            try:
                jsonObj = item.model_dump()
            except AttributeError:
                # Pydantic version below 2.0
                jsonObj = item.dict()

            formatted_elements.append(
                json.dumps(jsonObj, indent=4).replace("\n", "\n    ")
            )
        else:
            formatted_elements.append(repr(item))  # Fallback for other types

    formatted_list = ",\n    ".join(formatted_elements)
    return f"[\n    {formatted_list}\n]"


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print(
                "Event loop is already running. Applying nest_asyncio patch to allow async execution..."
            )
            nest_asyncio.apply()

        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def print_verbose_logs(metric: str, logs: str):
    print("*" * 50)
    print(f"{metric} Verbose Logs")
    print("*" * 50)
    print("")
    print(logs)
    print("")
    print("=" * 70)


def construct_verbose_logs(metric: CustomScorer, steps: List[str]) -> str:
    verbose_logs = ""
    for i in range(len(steps) - 1):
        verbose_logs += steps[i]

        # don't add new line for penultimate step
        if i < len(steps) - 2:
            verbose_logs += " \n \n"

    if metric.verbose_mode:
        # only print reason and score for judgeval
        print_verbose_logs(metric.__name__, verbose_logs + f"\n \n{steps[-1]}")

    return verbose_logs


def trimAndLoadJson(
    input_string: str, metric: Optional[CustomScorer] = None
) -> Any:
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
    # Remove trailing comma if one is present
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        if metric is not None:
            metric.error = error_str
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


class CustomFaithfulnessMetric(CustomScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        super().__init__(score_type="CUSTOM FAITHFULNESS", threshold=threshold)
        self.threshold = 1 if strict_mode else threshold
        self.using_native_model = True  # NOTE: SETTING THIS FOR LITELLM and TOGETHER usage
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def score_example(
        self,
        test_case: Example,
        all_claims: bool = False,
        _show_indicator: bool = True,
    ) -> float:

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_score_example(
                        test_case, 
                        all_claims=all_claims, 
                        _show_indicator=False
                    )
                )
            else:
                self.claims = self._generate_claims(test_case.actual_output, all_claims=all_claims)
                if self.additional_metadata is None:
                    self.additional_metadata = {}
                self.additional_metadata["claims"] = self.claims  # Add claims generated to metadata

                self.verdicts = self._generate_verdicts(test_case.retrieval_context)
                self.additional_metadata["verdicts"] = [v.model_dump() for v in self.verdicts]  # Add verdicts generated to metadata
                
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Claims:\n{prettify_list(self.claims)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_score_example(
        self,
        test_case: Example,
        all_claims: bool = False,
        _show_indicator: bool = True,
    ) -> float:
        
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            
            self.claims = await self._a_generate_claims(test_case.actual_output, all_claims=all_claims)
            if self.additional_metadata is None:
                self.additional_metadata = {}
            self.additional_metadata["claims"] = self.claims  # Add claims generated to metadata

            self.verdicts = await self._a_generate_verdicts(test_case.retrieval_context)  
            self.additional_metadata["verdicts"] = [v.model_dump() for v in self.verdicts]  # Add verdicts generated to metadata

            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Claims:\n{prettify_list(self.claims)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = FaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = FaithfulnessTemplate.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res = self.model.generate(prompt)
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    # @observe
    async def _a_generate_verdicts(self, retrieval_context: str) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []
        prompt = langfuse.get_prompt("FAITHFULNESS_REPLACEMENT")

        # Extract just the claims from the self.claims object
        claims = [claim["claim"] for claim in self.claims]

        prompt = prompt.compile(claims=claims, retrieval_context=retrieval_context)

        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = trimAndLoadJson(res, self)
            verdicts = [
                FaithfulnessVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(self, retrieval_context: str) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []
        prompt = langfuse.get_prompt("FAITHFULNESS_REPLACEMENT")
        # Extract just the claims from the self.claims object
        claims = [claim["claim"] for claim in self.claims]
        prompt = prompt.compile(claims=claims, retrieval_context=retrieval_context)
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = trimAndLoadJson(res, self)
            verdicts = [
                FaithfulnessVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    # @observe
    async def _a_generate_claims(self, actual_output: str, all_claims: bool = False) -> List[str]:
        if all_claims:
            prompt = langfuse.get_prompt("CLAIM_GENERATION", version=2)
        else:
            prompt = langfuse.get_prompt("CLAIM_GENERATION")
        prompt = prompt.compile(text=actual_output)
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = trimAndLoadJson(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = await self.model.a_generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _generate_claims(self, actual_output: str, all_claims: bool = False) -> List[str]:
        if all_claims:
            prompt = langfuse.get_prompt("CLAIM_GENERATION", version=2)  # update this
        else:
            prompt = langfuse.get_prompt("CLAIM_GENERATION")
        prompt = prompt.compile(text=actual_output)
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = trimAndLoadJson(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = self.model.generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        faithfulness_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

        score = faithfulness_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def _success_check(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    def get_claims(self):
        return self.claims
    
    def get_verdicts(self):
        return self.verdicts

    @property
    def __name__(self):
        return "Custom Faithfulness"

async def example():
    
    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    )

    example2 = Example(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        name="Password Reset",
        context=["User Account"],
        retrieval_context=["Password reset instructions"],
        tools_called=["authentication"],
        expected_tools=["authentication"],
        additional_metadata={"difficulty": "medium"}
    )

    model = TogetherJudge()

    scorer = CustomFaithfulnessMetric(threshold=0.5, 
                                      model=model)
    print(scorer)

    results = await a_execute_scoring(
        [example1, example2], 
        [scorer],
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=True,
        use_cache=False,
        throttle_value=0,
        max_concurrent=100,
    )
    pprint.pprint(results)


if __name__ == "__main__":
    res = asyncio.run(example())
