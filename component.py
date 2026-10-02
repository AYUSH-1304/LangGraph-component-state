import os
import operator

from typing import TypedDict, Annotated

from dotenv import load_dotenv, find_dotenv

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, END


# =========================================================
# 1. Load Environment Variables
# =========================================================

load_dotenv(find_dotenv())

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in .env file")


# =========================================================
# 2. Tavily Search Tool
# =========================================================

tool = TavilySearch(
    max_results=2
)

print("Tool Type:", type(tool))
print("Tool Name:", tool.name)


# =========================================================
# 3. Agent State
# =========================================================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# =========================================================
# 4. Agent
# =========================================================

class Agent:

    def __init__(self, model, tools, system=""):

        self.system = system

        self.tools = {
            t.name: t
            for t in tools
        }

        self.model = model.bind_tools(tools)

        # Track already-used searches
        self.used_queries = set()

        # Maximum number of tool calls per query
        self.max_tool_calls = 3

        self.tool_call_count = 0

        # -------------------------------------------------
        # Create LangGraph
        # -------------------------------------------------

        graph = StateGraph(AgentState)

        graph.add_node(
            "llm",
            self.call_llm
        )

        graph.add_node(
            "action",
            self.take_action
        )

        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {
                True: "action",
                False: END
            }
        )

        graph.add_edge(
            "action",
            "llm"
        )

        graph.set_entry_point("llm")

        self.graph = graph.compile()


    # =====================================================
    # Check Tool Calls
    # =====================================================

    def exists_action(self, state: AgentState):

        last_message = state["messages"][-1]

        tool_calls = getattr(
            last_message,
            "tool_calls",
            []
        )

        # No tool call
        if not tool_calls:
            return False

        # Stop if maximum tool calls reached
        if self.tool_call_count >= self.max_tool_calls:
            return False

        return True


    # =====================================================
    # Call LLM
    # =====================================================

    def call_llm(self, state: AgentState):

        messages = state["messages"]

        if self.system:

            messages = [
                SystemMessage(
                    content=self.system
                )
            ] + messages

        response = self.model.invoke(
            messages
        )

        return {
            "messages": [response]
        }


    # =====================================================
    # Execute Tools
    # =====================================================

    def take_action(self, state: AgentState):

        last_message = state["messages"][-1]

        tool_calls = getattr(
            last_message,
            "tool_calls",
            []
        )

        results = []

        for call in tool_calls:

            # Stop after maximum calls
            if self.tool_call_count >= self.max_tool_calls:
                break

            tool_name = call["name"]
            tool_args = call["args"]
            tool_id = call["id"]

            query = tool_args.get(
                "query",
                ""
            ).strip().lower()

            print("\n" + "-" * 60)

            print(
                "Calling Tool:",
                tool_name
            )

            print(
                "Arguments:",
                tool_args
            )

            # -------------------------------------------------
            # Prevent duplicate search
            # -------------------------------------------------

            if query in self.used_queries:

                result = (
                    "This search query was already executed. "
                    "Use the previous search results and "
                    "answer the user. Do not search again."
                )

                print(
                    "Duplicate query blocked."
                )

            else:

                self.used_queries.add(query)

                try:

                    result = self.tools[
                        tool_name
                    ].invoke(tool_args)

                    self.tool_call_count += 1

                except Exception as e:

                    result = (
                        f"Tool error: {e}"
                    )

            print(
                "Tool Result:",
                result
            )

            results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
            )

        print("-" * 60)

        return {
            "messages": results
        }


# =========================================================
# 5. System Prompt
# =========================================================

prompt = """
You are a smart research assistant.

You have access to a web search tool.

IMPORTANT RULES:

1. Use the search tool only when external/current
   information is required.

2. Search only when necessary.

3. Do NOT repeat the same search query.

4. After receiving useful search results,
   answer the user immediately.

5. Do NOT search again just because the first
   result is imperfect.

6. For multiple cities or subjects, you may
   perform separate searches, but keep searches
   to the minimum required.

7. Never get stuck in a tool-call loop.

8. Give a concise final answer.

9. If search results are incomplete or unreliable,
   clearly say that instead of repeatedly searching.
"""


# =========================================================
# 6. Groq Model
# =========================================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key,
)


# =========================================================
# 7. Create Agent
# =========================================================

abot = Agent(
    model=model,
    tools=[tool],
    system=prompt,
)


# =========================================================
# 8. Helper Function
# =========================================================

def run_query(query):

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)

    print(query)

    # Reset limits for every new question
    abot.used_queries.clear()
    abot.tool_call_count = 0

    messages = [
        HumanMessage(
            content=query
        )
    ]

    try:

        result = abot.graph.invoke(
            {
                "messages": messages
            }
        )

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(
            result["messages"][-1].content
        )

    except Exception as e:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)

        print(e)


# =========================================================
# 9. TEST 1
# =========================================================

run_query(
    "What is the weather in San Francisco?"
)


# =========================================================
# 10. TEST 2
# =========================================================

run_query(
    "What is the weather in San Francisco and Los Angeles?"
)


# =========================================================
# 11. TEST 3
# =========================================================

run_query(
    """
    Who won the Super Bowl in 2024?
    What is the GDP of the state that team represents?
    """
)