LangGraph Component & State

A simple **LangGraph AI Agent** using **Groq** and **Tavily Search**.

## Features

* LangGraph `StateGraph`
* Groq LLM
* Tavily web search
* Tool calling
* Duplicate search protection
* Maximum tool-call limit

## Install

```bash
pip install langgraph langchain-core langchain-groq langchain-tavily python-dotenv
```

## `.env`

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Run

```bash
python component.py
```

## Flow

```text
User → LLM → Tool → Tool Result → LLM → Final Answer
```

## Test Questions

* What is the weather in San Francisco?
* Compare weather in San Francisco and Los Angeles.
* Who won Super Bowl 2024?
* What was Missouri's GDP in 2023?


