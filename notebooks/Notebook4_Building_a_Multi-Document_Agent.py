# %% [markdown]
# Source notebook: Notebook4_Building_a_Multi-Document_Agent.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 4: Building a Multi-Document Agent

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
# ## 1. Setup an agent over 3 papers

# %% [markdown]
# **Note**: The pdf files are included with this lesson. To access these papers, go to the `File` menu and select`Open...`.

# %%
# Notebook cell 7
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]

# %%
# Notebook cell 8
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# %%
# Notebook cell 9
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# %%
# Notebook cell 10
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

# %%
# Notebook cell 11
len(initial_tools)

# %%
# Notebook cell 12
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# %%
# Notebook cell 13
response = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)

# %%
# Notebook cell 14
response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))

# %% [markdown]
# ## 2. Setup an agent over 11 papers

# %% [markdown]
# ### Download 11 ICLR papers

# %%
# Notebook cell 17
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=9WD9KwssyT",
    "https://openreview.net/pdf?id=yV6fD7LYkF",
    "https://openreview.net/pdf?id=hnrB5YHoYu",
    "https://openreview.net/pdf?id=WbWtOYIzIK",
    "https://openreview.net/pdf?id=c5pwL0Soay",
    "https://openreview.net/pdf?id=TpD2aG1h0D"
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf",
    "swebench.pdf",
    "selfrag.pdf",
    "zipformer.pdf",
    "values.pdf",
    "finetune_fair_diffusion.pdf",
    "knowledge_card.pdf",
    "metra.pdf",
    "vr_mcl.pdf"
]

# %% [markdown]
# To download these papers, below is the needed code:
# 
# 
#     #for url, paper in zip(urls, papers):
#          #!wget "{url}" -O "{paper}"
#     
#     
# **Note**: The pdf files are included with this lesson. To access these papers, go to the `File` menu and select`Open...`.

# %%
# Notebook cell 19
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# %% [markdown]
# ### Extend the Agent with Tool Retrieval

# %%
# Notebook cell 21
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# %%
# Notebook cell 22
# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

# %%
# Notebook cell 23
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# %%
# Notebook cell 24
tools = obj_retriever.retrieve(
    "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
)

# %%
# Notebook cell 25
tools[2].metadata

# %%
# Notebook cell 26
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True
)
agent = AgentRunner(agent_worker)

# %%
# Notebook cell 27
response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
print(str(response))

# %%
# Notebook cell 28
response = agent.query(
    "Compare and contrast the LoRA papers (LongLoRA, LoftQ). "
    "Analyze the approach in each paper first. "
)
