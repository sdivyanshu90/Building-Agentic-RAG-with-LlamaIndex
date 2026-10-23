# %% [markdown]
# Source notebook: Notebook3_Building_an_Agent_Reasoning_Loop.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 3: Building an Agent Reasoning Loop

# %% [markdown]
# ## Setup

# %%
# Notebook cell 3
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

# %%
# Notebook cell 4
import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
# ## Load the data

# %% [markdown]
# To download this paper, below is the needed code:
# 
# #!wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
# 
# **Note**: The pdf file is included with this lesson. To access it, go to the `File` menu and select`Open...`.

# %% [markdown]
# ## Setup the Query Tools

# %%
# Notebook cell 8
from utils import get_doc_tools

vector_tool, summary_tool = get_doc_tools("metagpt.pdf", "metagpt")

# %% [markdown]
# ## Setup Function Calling Agent

# %%
# Notebook cell 10
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo", 
    temperature=0
)

# %%
# Notebook cell 11
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# %%
# Notebook cell 12
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

# %%
# Notebook cell 13
print(response.source_nodes[0].get_content(metadata_mode="all"))

# %%
# Notebook cell 14
response = agent.chat(
    "Tell me about the evaluation datasets used."
)

# %%
# Notebook cell 15
response = agent.chat(
    "Tell me the results over one of the above datasets."
)

# %% [markdown]
# ## Lower-Level: Debuggability and Control

# %%
# Notebook cell 17
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# %%
# Notebook cell 18
task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

# %%
# Notebook cell 19
step_output = agent.run_step(task.task_id)

# %%
# Notebook cell 20
completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
print(completed_steps[0].output.sources[0].raw_output)

# %%
# Notebook cell 21
upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]

# %%
# Notebook cell 22
step_output = agent.run_step(
    task.task_id, 
    input="What about how agents share information?"
)

# %%
# Notebook cell 23
step_output = agent.run_step(task.task_id)
print(step_output.is_last)

# %%
# Notebook cell 24
response = agent.finalize_response(task.task_id)

# %%
# Notebook cell 25
print(str(response))
