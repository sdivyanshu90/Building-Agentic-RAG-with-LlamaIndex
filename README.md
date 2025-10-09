<div align="center">
<img src="https://github.com/user-attachments/assets/e09c8582-1791-42fc-b171-86e424535c1f" alt="Banner for DeepLearning.AI" width="400"/>
<h1 align="center">Building Agentic RAG with LlamaIndex</h1>
<p align="center">
<em>A course by <a href="https://www.deeplearning.ai/">DeepLearning.AI</a> in collaboration with <a href="https://www.llamaindex.ai/">LlamaIndex</a></em>
</p>
</div>

## 📖 Introduction
This repository contains notes, concepts, and detailed explanations related to the course "Building Agentic RAG with LlamaIndex." The course provides a comprehensive guide to moving beyond standard Retrieval-Augmented Generation (RAG) pipelines into the realm of intelligent, autonomous agents.

Unlike a standard RAG pipeline—which is suitable for simple queries across a few documents—the agentic approach taught in this course adapts its strategy based on initial findings to enhance further data retrieval. You will learn to develop an autonomous research agent, significantly enhancing your ability to engage with and analyze complex data.

Throughout the modules, you will practice building agents capable of intelligently navigating, summarizing, and comparing information across multiple research papers from arXiv. A key component of the course is learning how to effectively debug these agents, allowing you to inspect their reasoning and guide their actions to ensure accurate and reliable performance. This marks a critical step from simple data retrieval to sophisticated, multi-step data interaction and synthesis.

## 📚 Course Topics
This section provides an in-depth explanation of the core concepts covered in each module of the course.

<details>
<summary><strong>1. Router Query Engine: The First Step to Agentic Behavior</strong></summary>

What is a Router Query Engine?
The Router Query Engine represents the most fundamental form of an "agent" in an agentic RAG system. At its core, a router is a decision-making component that intelligently directs a user's query to the most appropriate processing pipeline. In a standard RAG setup, every query, regardless of its nature, is treated the same way: it is embedded, used to retrieve relevant text chunks, and then fed to a Large Language Model (LLM) for synthesis. This one-size-fits-all approach can be inefficient and may produce suboptimal results for queries with different intents. The Router Query Engine solves this problem by first analyzing the user's query and then selecting one of several specialized "downstream" query engines to handle it.

This module introduces the concept by building a router that must choose between two distinct engines: a Q&A (Question-Answering) Engine and a Summarization Engine. This simple yet powerful setup demonstrates the core principle of agentic RAG: choosing the right tool for the job.

How It Works: Intent Analysis and Selection
The "magic" of the router lies in its use of an LLM to understand the semantics and intent behind a user's query. The process works as follows:

Defining Tools: Each potential path—in this case, the Q&A engine and the summarization engine—is defined as a "tool" with clear metadata. For example:

Q&A Engine Tool: Its description might be, "This tool is best for answering specific, fact-based questions about the contents of the document."

Summarization Engine Tool: Its description could be, "This tool is best for providing a high-level, concise summary of the entire document or a large section of it."

Query Analysis: When a user submits a query, it is not immediately executed. Instead, the query, along with the descriptions of the available tools, is passed to a controlling LLM.

Decision Making: The LLM evaluates the query against the tool descriptions.

If the query is "What was the validation accuracy of the final model?", the LLM will recognize that this is a specific, fact-based question and determine that the Q&A Engine is the most suitable tool.

If the query is "Can you give me an overview of the paper's methodology?", the LLM will identify the user's intent to get a broad summary and select the Summarization Engine.

Execution: Once the LLM makes its selection, the router directs the original query to the chosen engine, which then processes it accordingly. The Q&A engine will perform a precise retrieval and synthesis to provide a direct answer, while the summarization engine will process the document to generate a comprehensive overview.

Why It's a Foundational Agentic Step
The Router Query Engine is more than just a conditional switch; it is the first step towards building an autonomous system. It embodies the principle of selection, where the system makes a reasoned choice based on context. This is a departure from a static, predetermined pipeline. By building a router, you learn to structure your RAG system as a collection of specialized capabilities and to use an LLM as an intelligent orchestrator to deploy those capabilities. This foundational concept of providing the LLM with a choice of tools is the essential building block for the more advanced agentic behaviors explored in subsequent modules, such as multi-step reasoning and dynamic tool use. It shifts the paradigm from simple retrieval to intelligent delegation.

</details>

<details>
<summary><strong>2. Tool Calling: Empowering Agents to Act</strong></summary>

From Selection to Action
The second module, "Tool Calling," represents a significant leap in capability from the simple router. While a router selects a pre-defined query engine, a tool-calling agent takes this a step further: it uses an LLM to not only select a function (a "tool") from a set of available options but also to infer and populate the arguments required to execute that function. This transforms the agent from a passive selector into an active participant that can perform concrete actions based on the user's natural language query. It allows the RAG system to break free from the confines of text retrieval and interact with external systems, APIs, or custom code.

This is a critical evolution. Instead of just deciding where a query should go, the agent now decides what action to perform and how to perform it. This is analogous to a human assistant who, when asked to "find out the weather in London," knows to open a weather app (the tool) and type "London" into the search bar (the argument).

The Mechanism of Tool Calling
The tool-calling process involves a sophisticated dialogue between the user, the LLM, and the application's codebase:

Tool Definition and Exposure: A developer defines a set of functions in their code that they want the agent to be able to use. These are the "tools." For each function, it is crucial to provide a detailed and descriptive docstring. This docstring serves as the metadata that the LLM will use to understand the tool's purpose, its required parameters (including their names and data types), and what it returns. LlamaIndex provides wrappers like FunctionTool to easily expose these Python functions to the agent.

LLM as Orchestrator: The user issues a query. The agent, powered by an LLM, receives this query along with the descriptions (docstrings) of all available tools.

Inference of Intent, Tool, and Arguments: The LLM performs a complex inference task on the user's query. It aims to determine:

Intent: Does this query require an external action, or can it be answered directly?

Tool Selection: If an action is needed, which of the available tools is the most appropriate for this task?

Argument Extraction: What specific values from the user's query should be used as arguments for the selected tool?

For instance, if a tool is defined as get_stock_price(ticker: str) and the user asks, "What is the current price of NVDA?", the LLM will identify the intent to get financial data, select the get_stock_price tool, and extract "NVDA" as the value for the ticker argument.

Structured Output and Execution: The LLM's decision is typically returned in a structured format, like JSON (e.g., {"tool": "get_stock_price", "arguments": {"ticker": "NVDA"}}). The application code parses this output, calls the corresponding Python function with the provided arguments, and captures the result (e.g., the stock price returned by an API).

Synthesizing the Final Response: The result from the tool's execution is then passed back to the LLM. The LLM uses this new piece of information to formulate a final, coherent, and human-readable answer for the user (e.g., "The current stock price for NVDA is $120.50.").

In this module, this process is explored in a "single-shot" setting, where the agent makes one tool call to answer a query. This foundational skill is the prerequisite for building more complex agents that can chain multiple tool calls together in a reasoning loop.

</details>

<details>
<summary><strong>3. Building an Agent Reasoning Loop: Enabling Multi-Step Thought</strong></summary>

Beyond Single-Shot Actions
The third module introduces the core concept that makes an agent truly intelligent and autonomous: the Agent Reasoning Loop. While the previous module focused on single-shot tool calling, most complex problems cannot be solved in a single step. A research assistant, for example, doesn't just find one piece of information; it formulates a plan, gathers multiple pieces of evidence, reflects on its findings, and synthesizes them into a comprehensive answer. The reasoning loop enables an agent to mimic this iterative process.

This is often implemented using a framework like ReAct (Reason, Act), which empowers an agent to cycle through a sequence of Thought → Action → Observation. This loop allows the agent to break down a complex user query into a series of smaller, manageable steps. It can maintain a state, learn from the results of its previous actions, and adjust its plan accordingly until it has gathered enough information to fully address the original request.

The Anatomy of the Reasoning Loop
The ReAct framework structures the agent's problem-solving process into a clear, iterative cycle:

Query Decomposition (Initial Thought): The agent receives a complex query, such as, "Compare the methodologies and performance results of the models presented in paper X." The agent's first step is to think and create a sub-goal. The LLM, acting as the agent's brain, might generate an internal monologue like: "I need to solve this in multiple steps. First, I will find and summarize the methodology section of the paper. Then, I will find the performance results."

Action: Based on its thought, the agent selects a tool and decides on its arguments. This could be a tool called retrieve_section(section_name="Methodology"). The agent then executes this action.

Observation: The agent observes the outcome of its action. The retrieve_section tool returns the text of the methodology section. This result is the "Observation."

Iteration (New Thought): The agent now enters the next cycle of the loop. It reflects on its initial goal, its previous action, and the new observation. Its internal state, often called a "scratchpad," now contains the query and the retrieved methodology. The agent's next thought might be: "I have the methodology. Now I need to find the performance results. I will use the tool retrieve_section(section_name="Results")."

Further Action and Observation: The agent executes this new action and observes the results—the text from the results section.

Final Synthesis (Final Thought): The agent now has all the necessary components in its scratchpad: the original query, the methodology text, and the results text. It determines that it has gathered enough information. Its final thought is to synthesize all this information into a coherent answer that directly addresses the user's comparison query. It then generates the final, detailed response.

The Role of LlamaIndex and Debugging
LlamaIndex provides powerful abstractions, like the ReActAgent, which manage the entire reasoning loop. It handles maintaining the scratchpad, formatting the prompts for the LLM at each step (to include the history of thoughts, actions, and observations), parsing the LLM's output to distinguish between thoughts and actions, and dispatching actions to the correct tools.

Crucially, this module emphasizes debugging. Because the agent's reasoning is explicit (i.e., the "thoughts" are recorded), you can trace its decision-making process step-by-step. If the agent makes a mistake, you can inspect its thought process to understand why it chose a particular action, allowing you to refine the tool descriptions, the agent's prompt, or the tools themselves to improve performance. This transparency is key to building reliable and effective agents.

</details>

<details>
<summary><strong>4. Building a Multi-Document Agent: Scaling Agentic RAG</strong></summary>

The Challenge of Multiple Sources
The final module of the course addresses a critical real-world challenge: scaling the agentic system from a single document to a corpus of multiple documents. Most practical applications, especially in research or enterprise settings, require an agent to reason across and synthesize information from numerous sources. A single-document agent knows where to look for information; a multi-document agent must first figure out which document(s) contain the relevant information before it can even begin to extract and analyze it. This adds a crucial layer of retrieval and selection to the agent's reasoning process.

This module teaches you how to extend the agentic reasoning loop to build a powerful research assistant that can intelligently navigate, compare, and synthesize findings from a collection of different research papers.

Architecture of a Multi-Document Agent
To handle multiple documents, the architecture of the RAG system must evolve. Instead of a single knowledge base, the system now manages multiple, distinct knowledge bases, often one for each document. The key is to equip the agent with tools that allow it to interact with this collection of sources.

The process typically involves these components:

Indexing Per Document: Each document in the corpus (e.g., each arXiv paper) is indexed separately. For example, you might create a dedicated VectorStoreIndex for each paper. This compartmentalizes the knowledge and allows for targeted querying.

Creating Document-Specific Tools: Each document's index is then wrapped into a QueryEngineTool. This creates a specific tool for querying each document. The description for each tool is vital for the agent's success; it should contain metadata about the document, such as its title, authors, and a concise summary of its abstract. This metadata allows the agent to infer which document is likely to contain the answer to a specific question.

The Agent's Multi-Document Reasoning Loop: The agent's reasoning process is now enhanced to include a document selection step. When faced with a query like, "Contrast the data augmentation techniques used in 'Paper A' with the ones in 'Paper B'," the agent's loop might look like this:

Thought: "The user wants to compare two papers. I need to get information from 'Paper A' first."

Action (Tool Selection): The agent examines the descriptions of all its document-querying tools. Based on the titles in the descriptions, it selects the tool corresponding to 'Paper A' and formulates a sub-query for it, such as "What data augmentation techniques were used?".

Observation: The agent receives the answer from querying 'Paper A'. This information is added to its scratchpad.

Thought: "I have the information for Paper A. Now I need to do the same for 'Paper B'."

Action: The agent selects the tool for 'Paper B' and asks the same sub-query.

Observation: The agent receives the information from 'Paper B'.

Final Synthesis: With the necessary information from both documents now in context, the agent performs the final step of comparing the two sets of techniques and generating a comprehensive answer for the user.

This approach combines all the concepts from the previous modules: the agent routes its query to the correct document (like the Router), calls a tool to query that document, and does so within a reasoning loop to handle the multi-step nature of the comparative query. This creates a highly capable and autonomous research agent that can perform complex analyses across an entire library of documents.

</details>

## 🙏 Acknowledgement
This course is made possible by the incredible teams at **[DeepLearning.AI](https://www.deeplearning.ai/)** and **[LlamaIndex](https://www.llamaindex.ai/)**. Their collaboration has produced a clear, practical, and powerful curriculum for anyone looking to advance their skills in building the next generation of LLM applications.
