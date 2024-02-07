# Model Techniques
***Level up your responses***

OK so you never heard about MLflow but know about langchain. How does all this fit together? Let's first walk through how people use LLMs in practice:

## Prompt Templating

## RAG: Retrieval Augmented Generation

## ReACT: Agents

## Function Calling

## Streaming State Summaries
* run a local embedding service (AF)
* concat events from bus into state window
    * ring buffer
* embed ring buffer
    * store in TSDB? what is TSDB + VectorDB? Why, simply a TSDB where the values are embeddings of course! So we need a separable similarity score/search API over a time range query
* summarize states within window (sliding window summaries)
* streaming RAG uses the embedded ring buffer to know "what's happening"
* 1. ask LLM what time range to look up (function calling/REACT)
* 2. extract set(embeddings) -> (now ingest into local FAISS etc to simplify search API?)
* 3. search for similar embeddings to retrieve related events
* 4. summarize events via LLM
