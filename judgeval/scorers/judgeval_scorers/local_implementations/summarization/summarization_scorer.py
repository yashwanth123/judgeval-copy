from typing import List, Optional, Union
import asyncio
from judgment import *

from judgeval.scorers.utils import (get_or_create_event_loop,
                                    scorer_progress_meter,
                                    create_verbose_logs,
                                    parse_response_json,
                                    check_example_params
                                    )
from judgeval.scorers import CustomScorer
from judgeval.judges import judgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers.judgeval_scorers.local_implementations.faithfulness.prompts import (
    FaithfulnessTemplate, 
    Claims
)
from judgeval.scorers.judgeval_scorers.local_implementations.summarization.prompts import *


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
]


class SummarizationMetric(CustomScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        n: int = 5,
        model: Optional[Union[str, judgevalJudge]] = None,
        assessment_questions: Optional[List[str]] = None,
        include_reason: bool = True,
        async_mode=True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        user: Optional[str] = None
    ):
        # logger.info(f"Initializing SummarizationMetric with model: {model}")
        self.user = user
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = create_judge(model, user=user)
        self.evaluation_model = self.model.get_model_name()

        if assessment_questions is not None and len(assessment_questions) == 0:
            self.assessment_questions = None
        else:
            self.assessment_questions = assessment_questions

        self.include_reason = include_reason
        self.n = n
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

        # logger.debug(f"Initialized with threshold: {self.threshold}, n: {n}, strict_mode: {strict_mode}")

    def score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        # logger.info("Starting synchronous summarization measurement")
        check_example_params(example, required_params, self)
        
        with scorer_progress_meter(self, display_meter=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_score_example(example, _show_indicator=False)
                )
            else:
                self.claims: List[str] = self._generate_claims(
                    example.actual_output
                )
                # logger.debug(f"Generated {len(self.claims)} claims")
                
                self.coverage_verdicts: List[SummarizationCoverageVerdict] = (
                    self._generate_coverage_verdicts(example)
                )
                # logger.debug(f"Generated {len(self.coverage_verdicts)} coverage verdicts")
                
                self.alignment_verdicts: List[SummarizationAlignmentVerdict] = (
                    self._generate_alignment_verdicts(example)
                )
                # logger.debug(f"Generated {len(self.alignment_verdicts)} alignment verdicts")
                
                alignment_score = self._calculate_score(ScoreType.ALIGNMENT)
                coverage_score = self._calculate_score(ScoreType.COVERAGE)
                self.score_breakdown = {
                    ScoreType.ALIGNMENT.value: alignment_score,
                    ScoreType.COVERAGE.value: coverage_score,
                }
                self.score = min(alignment_score, coverage_score)
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Claims:\n{self.claims}",
                        f"Assessment Questions:\n{self.assessment_questions}",
                        f"Coverage Verdicts:\n{[v.model_dump() for v in self.coverage_verdicts]}",
                        f"Alignment Verdicts:\n{[v.model_dump() for v in self.alignment_verdicts]}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                # logger.info(f"Measurement complete. Score: {self.score}, Success: {self.success}")
                return self.score

    async def a_score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        """
        To score, we take the following steps:
        1. Generate claims from the actual output
            - Extract key factual claims from the summary text

        2. Generate coverage verdicts:
            a. Generate assessment questions if not provided
            b. Generate answers to the assessment questions for both summary and original text
            c. Compare answers to determine if summary adequately covers key information
            d. Calculate coverage score based on matching answers

        3. Generate alignment verdicts:
            a. Generate claims from the actual output
            b. Verify each claim against the original text for factual accuracy
            c. Calculate alignment score based on verified claims

        4. Calculate final score:
            - Take minimum of coverage and alignment scores
            - Generate reason explaining the scoring
            - Check if score meets threshold for success
        """
        # logger.debug("Starting async measurement")
        check_example_params(example, required_params, self)

        with scorer_progress_meter(
            self,
            async_mode=True,
            display_meter=_show_indicator,
        ):
            self.claims = await self._a_generate_claims(example.actual_output),
            
            self.coverage_verdicts, self.alignment_verdicts = await asyncio.gather(
                self._a_generate_coverage_verdicts(example),
                self._a_generate_alignment_verdicts(example),
            )
            # logger.debug(f"Generated {len(self.coverage_verdicts)} coverage and {len(self.alignment_verdicts)} alignment verdicts")
            
            alignment_score = self._calculate_score(ScoreType.ALIGNMENT)
            coverage_score = self._calculate_score(ScoreType.COVERAGE)
            self.score_breakdown = {
                ScoreType.ALIGNMENT.value: alignment_score,
                ScoreType.COVERAGE.value: coverage_score,
            }
            self.score = min(alignment_score, coverage_score)
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Claims:\n{self.claims}",
                    f"Assessment Questions:\n{self.assessment_questions}",
                    f"Coverage Verdicts:\n{[v.model_dump() for v in self.coverage_verdicts]}",
                    f"Alignment Verdicts:\n{[v.model_dump() for v in self.alignment_verdicts]}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            # logger.info(f"Async measurement complete. Score: {self.score}, Success: {self.success}")
            return self.score

    async def _a_generate_reason(self) -> str:
        # logger.debug("Generating reason asynchronously")
        if self.include_reason is False:
            # logger.debug("Reason generation skipped - include_reason is False")
            return None

        contradictions = []
        redundancies = []
        for verdict in self.alignment_verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            elif verdict.verdict.strip().lower() == "idk":
                redundancies.append(verdict.reason)

        questions = []
        if self.coverage_verdicts:
            for verdict in self.coverage_verdicts:
                if (
                    verdict.original_verdict.strip().lower() == "yes"
                    and verdict.summary_verdict.strip().lower() == "no"
                ):
                    questions.append(verdict.question)

        prompt: dict = SummarizationTemplate.generate_reason(
            contradictions=contradictions,
            redundancies=redundancies,
            questions=questions,
            score=format(self.score, ".2f"),
        )

        if len(questions) > 0:
            prompt += f"""Questions the original text can answer but not the summary:
{questions}

"""
        prompt += """JSON:
"""

        if self.using_native_model:
            res = await self.model.a_generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason, user=self.user)
                return res.reason
            except TypeError:
                # logger.error("Failed to parse response with schema, falling back to raw json parsing")
                res = await self.model.a_generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        redundancies = []
        for verdict in self.alignment_verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            elif verdict.verdict.strip().lower() == "idk":
                redundancies.append(verdict.reason)

        questions = []
        if self.coverage_verdicts:
            for verdict in self.coverage_verdicts:
                if (
                    verdict.original_verdict.strip().lower() == "yes"
                    and verdict.summary_verdict.strip().lower() == "no"
                ):
                    questions.append(verdict.question)

        prompt: dict = SummarizationTemplate.generate_reason(
            contradictions=contradictions,
            redundancies=redundancies,
            questions=questions,
            score=format(self.score, ".2f"),
        )

        if len(questions) > 0:
            prompt += f"""Questions the original text can answer but not the summary:
{questions}

"""
        prompt += """JSON:
"""

        if self.using_native_model:
            res = self.model.generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason, user=self.user)
                return res.reason
            except TypeError:
                # logger.error("Failed to parse response with schema, falling back to raw json parsing")
                res = self.model.generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                return data["reason"]

    def _calculate_score(self, score_type: ScoreType) -> float:
        # logger.debug(f"Calculating {score_type.value} score")
        if score_type == ScoreType.ALIGNMENT:
            total = len(self.alignment_verdicts)
            if total == 0:
                # logger.warning("No alignment verdicts found, returning score of 0")
                return 0
            faithfulness_count = 0
            for verdict in self.alignment_verdicts:
                # Different from the faithfulness score, this
                # penalizes 'idk' (full of fluff) summaries
                if verdict.verdict.strip().lower() == "yes":
                    faithfulness_count += 1

            score = faithfulness_count / total

        else:
            if self.assessment_questions is None:
                # logger.debug("No assessment questions provided, returning perfect coverage score")
                return 1
            total = 0
            coverage_count = 0
            for verdict in self.coverage_verdicts:
                if verdict.original_verdict.strip().lower() == "yes":
                    total += 1
                    if verdict.summary_verdict.strip().lower() == "yes":
                        coverage_count += 1

            if total == 0:
                # logger.warning("No valid coverage verdicts found, returning score of 0")
                return 0

            score = coverage_count / total

        # logger.debug(f"Calculated {score_type.value} score: {score}")
        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_answers(self, text: str) -> List[str]:
        prompt = SummarizationTemplate.generate_answers(
            questions=self.assessment_questions, text=text
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            return data["answers"]
        else:
            try:
                res: Answers = await self.model.a_generate(
                    prompt, schema=Answers
                )
                return res.answers
            except TypeError:
                res = await self.model.a_generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                return data["answers"]

    def _generate_answers(self, text: str) -> List[str]:
        # logger.debug("Generating answers")
        prompt = SummarizationTemplate.generate_answers(
            questions=self.assessment_questions, text=text
        )
        if self.using_native_model:
            res = self.model.generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            return data["answers"]
        else:
            try:
                res: Answers = self.model.generate(prompt, schema=Answers, user=self.user)
                return res.answers
            except TypeError:
                # logger.error("Failed to parse answers with schema, falling back to raw json parsing")
                res = self.model.generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                return data["answers"]

    async def _a_generate_assessment_questions(self, text: str):
        prompt = SummarizationTemplate.generate_questions(text=text, n=self.n)
        if self.using_native_model:
            res = await self.model.a_generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            return data["questions"]
        else:
            try:
                res: Questions = await self.model.a_generate(
                    prompt, schema=Questions
                )
                return res.questions
            except TypeError:
                res = await self.model.a_generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                return data["questions"]

    def _generate_assessment_questions(self, text: str):
        # logger.debug(f"Generating {self.n} assessment questions")
        prompt = SummarizationTemplate.generate_questions(text=text, n=self.n)
        if self.using_native_model:
            res = self.model.generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            return data["questions"]
        else:
            try:
                res: Questions = self.model.generate(prompt, schema=Questions, user=self.user)
                return res.questions
            except TypeError:
                # logger.error("Failed to parse questions with schema, falling back to raw json parsing")
                res = self.model.generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                return data["questions"]

    async def _a_generate_coverage_verdicts(
        self, test_case: Example
    ) -> List[SummarizationCoverageVerdict]:
        if self.assessment_questions is None:
            self.assessment_questions = (
                await self._a_generate_assessment_questions(test_case.input)
            )

        tasks = [
            self._a_generate_answers(test_case.input),
            self._a_generate_answers(test_case.actual_output),
        ]
        results = await asyncio.gather(*tasks)
        original_answers = results[0]
        summary_answers = results[1]

        if len(original_answers) != len(summary_answers):
            # logger.error(f"Mismatched answer lengths: original={len(original_answers)}, summary={len(summary_answers)}")
            raise ValueError("Number of verdicts generated does not equal.")

        coverage_veridcts: List[SummarizationCoverageVerdict] = []
        for i in range(len(original_answers)):
            coverage_veridcts.append(
                SummarizationCoverageVerdict(
                    summary_verdict=summary_answers[i],
                    original_verdict=original_answers[i],
                    question=self.assessment_questions[i],
                )
            )
        return coverage_veridcts

    def _generate_coverage_verdicts(
        self, test_case: Example
    ) -> List[SummarizationCoverageVerdict]:
        if self.assessment_questions is None:
            self.assessment_questions = self._generate_assessment_questions(
                test_case.input
            )

        original_answers = self._generate_answers(test_case.input)
        summary_answers = self._generate_answers(test_case.actual_output)

        if len(original_answers) != len(summary_answers):
            # logger.error(f"Mismatched answer lengths: original={len(original_answers)}, summary={len(summary_answers)}")
            raise ValueError("Number of verdicts generated does not equal.")

        coverage_veridcts: List[SummarizationCoverageVerdict] = []
        for i in range(len(original_answers)):
            coverage_veridcts.append(
                SummarizationCoverageVerdict(
                    summary_verdict=summary_answers[i],
                    original_verdict=original_answers[i],
                    question=self.assessment_questions[i],
                )
            )

        return coverage_veridcts

    async def _a_generate_alignment_verdicts(
        self,
        test_case: Example,
    ) -> List[SummarizationAlignmentVerdict]:
        if len(self.claims) == 0:
            # logger.warning("No claims generated, returning empty alignment verdicts")
            return []

        verdicts: List[SummarizationAlignmentVerdict] = []
        prompt = langfuse.get_prompt("ALIGNMENT_REPLACEMENT")
        prompt = prompt.compile(
            claims=self.claims,
            original=test_case.input
                       )
        if self.using_native_model:
            res = await self.model.a_generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            verdicts = [
                SummarizationAlignmentVerdict(**item)
                for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts, user=self.user
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                # logger.error("Failed to parse verdicts with schema, falling back to raw json parsing")
                res = await self.model.a_generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                verdicts = [
                    SummarizationAlignmentVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _generate_alignment_verdicts(
        self,
        test_case: Example,
    ) -> List[SummarizationAlignmentVerdict]:
        if len(self.claims) == 0:
            # logger.warning("No claims generated, returning empty alignment verdicts")
            return []

        verdicts: List[SummarizationAlignmentVerdict] = []

        prompt = langfuse.get_prompt("ALIGNMENT_REPLACEMENT")
        prompt = prompt.compile(
            claims=self.claims,
            original=test_case.input
                       )
        if self.using_native_model:
            res = self.model.generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            verdicts = [
                SummarizationAlignmentVerdict(**item)
                for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts, user=self.users)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                # logger.error("Failed to parse verdicts with schema, falling back to raw json parsing")
                res = self.model.generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                verdicts = [
                    SummarizationAlignmentVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    async def _a_generate_claims(self, text: str) -> List[str]:
        # Borrow faithfulness template
        prompt = FaithfulnessTemplate.generate_claims(text=text)
        if self.using_native_model:
            res = await self.model.a_generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = await self.model.a_generate(prompt, schema=Claims, user=self.user)
                return res.claims
            except TypeError:
                # logger.error("Failed to parse verdicts with schema, falling back to raw json parsing")
                res = await self.model.a_generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                return data["claims"]

    def _generate_claims(self, text: str) -> List[str]:
        # Borrow faithfulness template
        prompt = FaithfulnessTemplate.generate_claims(text=text)
        if self.using_native_model:
            res = self.model.generate(prompt, user=self.user)
            data = parse_response_json(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = self.model.generate(prompt, schema=Claims, user=self.user)
                return res.claims
            except TypeError:
                # logger.error("Failed to parse claims with schema, falling back to raw json parsing")
                res = self.model.generate(prompt, user=self.user)
                data = parse_response_json(res, self)
                return data["claims"]

    def success_check(self) -> bool:
        if self.error is not None:
            # logger.warning(f"Metric failed with error: {self.error}")
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
                # logger.debug(f"Success check: score {self.score} >= threshold {self.threshold} = {self.success}")
            except:
                # logger.error("Failed to determine success status", exc_info=True)
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Summarization"