from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.experimental.plan_and_execute import load_chat_planner
from plan_and_execute_retry import PlanAndExecute
from loading import load_agent_executor, run_step_eval

ai_working_dir = './working'

tk = FileManagementToolkit(root_dir=ai_working_dir)
tools = tk.get_tools()
llm = ChatOpenAI(model_name='gpt-4', streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
llm3 = ChatOpenAI(model_name='gpt-3.5-turbo', streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

SUFFIX = (
    "Begin! Reminder to ALWAYS respond with a valid json blob of a single action."
    " Remember, you MUST respond with a 'Final Answer' action to respond directly when you are completely finished. Do not stop responding until you have completed the task."
    " Use tools if necessary. Respond directly when you are completely finished, with as much context an detail as possible. Format is Action:```$JSON_BLOB```then Observation:."
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

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={'suffix': SUFFIX},
    verbose=True
)

tools_str = "\n".join([f"```\n{tool.name}\n{tool.description}\n```" for tool in tools])
SYSTEM_PROMPT = SYSTEM_PROMPT.format(tools=tools_str)
planner = load_chat_planner(llm=llm, system_prompt=SYSTEM_PROMPT)
executor = load_agent_executor(llm=llm, tools=tools, verbose=True)
eval_chain = lambda x, y, z : run_step_eval(llm, x, y, z)
agent_pande_chain = PlanAndExecute(planner=planner, executor=executor, eval_chain=eval_chain, verbose=True)

if __name__ == '__main__':
    #agent_chain.run('Create a new python program that can do something cool.')
    agent_pande_chain.run('Create a new python progam that can do something cool.')
