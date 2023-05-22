from typing import List, Any

from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.base_language import BaseLanguageModel
from langchain.experimental.plan_and_execute.executors.base import BaseExecutor
from langchain.tools import BaseTool
from langchain.callbacks.manager import Callbacks
from langchain.experimental.plan_and_execute.schema import StepResponse
from langchain.chains.base import Chain
from langchain import LLMChain
from langchain.schema import SystemMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.experimental.plan_and_execute.planners.base import LLMPlanner
from langchain.experimental.plan_and_execute.planners.chat_planner import PlanningOutputParser


HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{critique}
{agent_scratchpad}"""

PREFIX = (
    "Complete the task that the human gives you to the best of your ability."
    " Respond as accurately and helpfully as possible, and do NOT do anything outside the scope of your objective."
    " You are an AI language model that can make decisions and take actions on its own, without human intervention."
    " You must complete the task that the human gives you, and you must do so without any human intervention."
    " You have upgraded capabilities compared to other AI language models, and you have access to a set of tools that you can use to complete your objective."
    " You have access to the following tools:"
)

SUFFIX = (
    "Begin! Reminder to ALWAYS respond with a valid json blob of a single action."
    " Use tools if necessary. Do not do anything outside the scope of your current objective."
    " Respond directly only when you have completed your current objective. Once you use the Final Answer tool, you will not be able to use any other tools."
    " Format is Action:```$JSON_BLOB```then Observation:."
    "\nThought:"
)

SYSTEM_PROMPT = (
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "to accurately complete the task. This plan will be executed by an upgraded AI agent that can take actions and make decisions on its own."
    " The plan should be as detailed as possible, and should be able to be executed by an AI agent without any human intervention."
    " The plan should be specific, and should contain as much detail and as few ambiguities as possible."
    "\nIf the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "\nIn completing the plan, the agent will have access to a set of tools defined by the user. "
    "You will also see these tools, and can reference them when creating the plan."
)

class ChainExecutor(BaseExecutor):
    chain: Chain

    def step(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""
        response = self.chain(inputs, callbacks=callbacks)
        return (StepResponse(response=response['output']), response['intermediate_steps'])

    async def astep(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""
        response = await self.chain.arun(**inputs, callbacks=callbacks)
        return StepResponse(response=response)

def load_agent_executor(
    llm: BaseLanguageModel, tools: List[BaseTool], verbose: bool = False
) -> ChainExecutor:
    agent = StructuredChatAgent.from_llm_and_tools(
        llm,
        tools,
        human_message_template=HUMAN_MESSAGE_TEMPLATE,
        input_variables=["previous_steps", "current_step", "critique", "agent_scratchpad"],
        prefix=PREFIX,
        suffix=SUFFIX
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose, return_intermediate_steps=True
    )
    return ChainExecutor(chain=agent_executor)

def load_chat_planner(
    llm: BaseLanguageModel, tools: List[BaseTool], system_prompt: str = SYSTEM_PROMPT
) -> LLMPlanner:

    input = f"The agent will have access to the following tools when executing the plan:\n{tools}\n\n"
    input += "{input}"
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template(input),
        ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return LLMPlanner(
        llm_chain=llm_chain,
        output_parser=PlanningOutputParser(),
        stop=["<END_OF_PLAN>"],
    )

def load_step_evaluator(llm):
    sys_message = (
        "You are an intelligent evaluator that determines whether a given response successfully completes a step in a plan, or whether it needs to be retried."
        "\nYou only respond with '<yes>' if the response completes the task, and '<no>' followed by a critique otherwise. The critique will be used to help the agent improve its response."
    )
    human_message = (
        "Given this step in a plan: {step}, and the following response from the agent, does the response successfully complete the step?"
        " If so, respond with '<yes>'. If not, respond with '<no>' followed by a critique of the response."
        "\n\nResponse:\n{response}"
    )
    sysm = SystemMessagePromptTemplate.from_template(sys_message)
    humm = HumanMessagePromptTemplate.from_template(human_message)
    prompt = ChatPromptTemplate.from_messages([sysm, humm])

    return LLMChain(llm=llm, prompt=prompt)

def parse_eval_output(output):
    success = False
    critique = ""
    if output.startswith("<yes>"):
        success = True
    elif output.startswith("<no>"):
        critique = output[5:]
    else:
        raise ValueError(f"Invalid output: {output}")
    return {"success": success, "critique": critique}

def run_step_eval(llm, step, response, intermediate_steps):
    evaluator = load_step_evaluator(llm)
    intermediate_steps_str = ["\n".join(map(str, step)) for step in intermediate_steps]
    full_response = "\n\n".join(intermediate_steps_str + [response])
    inputs = {"step": step, "response": full_response}
    output = evaluator.run(inputs)
    return parse_eval_output(output)