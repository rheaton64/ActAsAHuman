from typing import Any, Dict, List, Optional, Callable
import copy

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.experimental.plan_and_execute.executors.base import BaseExecutor
from langchain.experimental.plan_and_execute.planners.base import BasePlanner
from langchain.experimental.plan_and_execute.schema import (
    BaseStepContainer,
    ListStepContainer,
)


class PlanAndExecute(Chain):
    planner: BasePlanner
    executor: BaseExecutor
    eval_chain: Optional[Callable] = None # Should lambda away the llm here
    step_container: BaseStepContainer = Field(default_factory=ListStepContainer)
    input_key: str = "input"
    output_key: str = "output"
    max_retries: int = 3

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        plan = self.planner.plan(
            inputs,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        if run_manager:
            run_manager.on_text(str(plan), verbose=self.verbose)
        for step in plan.steps:
            _new_inputs = {"previous_steps": self.step_container, "current_step": step, "critique": ""}
            new_inputs = {**_new_inputs, **inputs}
            response = None
            for i in range(self.max_retries):
                retry_inputs = copy.deepcopy(new_inputs)
                (response, int_steps) = self.executor.step(
                    new_inputs,
                    callbacks=run_manager.get_child() if run_manager else None,
                )
                if self.eval_chain:
                    eval_result = self.eval_chain(step.value, response.response, int_steps)
                    print(eval_result)
                else:
                    eval_result = {"success": True}
                if eval_result['success']:
                    break
                else:
                    if run_manager:
                        run_manager.on_text(
                            f"*****\n\nStep: {step.value} failed. Retry {i+1}/{self.max_retries}", verbose=self.verbose
                        )
                    # append the critique to the inputs for the retry
                    retry_inputs['critique'] = eval_result['critique']

                    if i == self.max_retries - 1:
                        raise Exception(f"Step: {step.value} failed. Max retries reached.")
                if run_manager:
                    run_manager.on_text(
                        f"*****\n\nStep: {step.value}", verbose=self.verbose
                    )
                    run_manager.on_text(
                        f"\n\nResponse: {response.response}", verbose=self.verbose
                    )
            self.step_container.add_step(step, response)
        return {self.output_key: self.step_container.get_final_response()}