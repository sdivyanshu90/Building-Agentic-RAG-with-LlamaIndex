# %% [markdown]
# Source notebook: Notebook1_Router_Engine.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 1: Router Engine

# %% [markdown]
# Welcome to Lesson 1.
# 
# To access the `requirements.txt` file, the data/pdf file required for this lesson and the `helper` and `utils` modules, please go to the `File` menu and select`Open...`.
# 
# I hope you enjoy this course!

# %% [markdown]
# ## Setup

# %%
# Notebook cell 4
from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

# %%
# Notebook cell 5
import nest_asyncio

nest_asyncio.apply()

# %% [markdown]
# ## Load Data

# %% [markdown]
# To download this paper, below is the needed code:
# 
# #!wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
# 
# **Note**: The pdf file is included with this lesson. To access it, go to the `File` menu and select`Open...`.

# %%
# Notebook cell 8
from llama_index.core import SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()

# %% [markdown]
# ## Define LLM and Embedding model

# %%
# Notebook cell 10
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# %%
# Notebook cell 11
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# %% [markdown]
# ## Define Summary Index and Vector Index over the Same Data

# %%
# Notebook cell 13
from llama_index.core import SummaryIndex, VectorStoreIndex

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# %% [markdown]
# ## Define Query Engines and Set Metadata

# %%
# Notebook cell 15
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

# %%
# Notebook cell 16
from llama_index.core.tools import QueryEngineTool


summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)

# %% [markdown]
# ## Define Router Query Engine

# %%
# Notebook cell 18
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

# %%
# Notebook cell 19
response = query_engine.query("What is the summary of the document?")
print(str(response))

# %%
# Notebook cell 20
print(len(response.source_nodes))

# %%
# Notebook cell 21
response = query_engine.query(
    "How do agents share information with other agents?"
)
print(str(response))

# %% [markdown]
# ## Let's put everything together

# %%
# Notebook cell 23
from utils import get_router_query_engine

query_engine = get_router_query_engine("metagpt.pdf")

# %%
# Notebook cell 24
response = query_engine.query("Tell me about the ablation study results?")
print(str(response))
