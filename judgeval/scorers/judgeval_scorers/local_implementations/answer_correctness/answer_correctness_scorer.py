from typing import Optional, List, Union
from pydantic import BaseModel

from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers import JudgevalScorer
from judgeval.scorers.utils import (
    get_or_create_event_loop,
    parse_response_json,
    scorer_progress_meter,
    create_verbose_logs,
    check_example_params,
)
from .prompts import ACVerdict, AnswerCorrectnessTemplate


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.EXPECTED_OUTPUT,
]


class AnswerCorrectnessScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, JudgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        user: Optional[str] = None
    ):
        self.user = user
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.model, self.using_native_model = create_judge(model, user=user)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def _a_get_statements(self, expected_output: str) -> List[str]:
        pass

    def _a_get_verdicts(self, actual_output: str) -> List[ACVerdict]:
        pass

    def _a_get_reason(self) -> str:
        pass

    def _compute_score(self) -> float:
        pass

    def score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)

        with scorer_progress_meter(self, display_meter=_show_indicator):
            try:
                if self.async_mode:
                    loop = get_or_create_event_loop()
                    loop.run_until_complete(
                        self.a_score_example(example, _show_indicator=False)
                    )
                else:
                    self.score = self._compute_score()  # TODO: replace this with the actual implementation
            except Exception:
                raise
    
    async def a_score_example(
        self, 
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)

        with scorer_progress_meter(self, async_mode=True, display_meter=_show_indicator):
            try:
                self.statements: List[str] = await self._a_get_statements(example.expected_output)
                self.verdicts: List[ACVerdict] = await self._a_get_verdicts(example.actual_output)
                self.score = self._compute_score()
                self.reason = await self._a_get_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Statements:\n{self.statements}",
                        f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            except Exception as e:
                print(f"Error in a_score_example for AnswerCorrectnessScorer: {e}")
                raise

    def success_check(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Answer Correctness"
