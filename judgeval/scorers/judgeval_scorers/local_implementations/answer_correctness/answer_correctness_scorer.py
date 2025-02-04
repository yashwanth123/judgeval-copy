from typing import Optional, List, Union

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

    def score_example(self, example: Example) -> float:
        return
    
    async def a_score_example(self, example: Example) -> float:
        return

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
