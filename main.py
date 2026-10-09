
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# State passed between graph nodes
class FlowState(TypedDict):
    question: str
    category: str
    answer: str


# Structured output for question classification
class QuestionCategory(BaseModel):
    category: Literal["coding", "weather", "google_search"] = Field(
        description="Choose the category that best matches the question."
    )


# Google Search Agent
search = GoogleSerperAPIWrapper(
    serper_api_key=os.getenv("SERPER_API_KEY")
)


@tool
def google_search(query: str) -> str:
    """Search Google for current information."""
    return search.run(query)


google_agent = create_agent(
    model=llm,
    tools=[google_search],
    system_prompt="Answer using Google search when needed. Be accurate.",
)


# Weather Agent: demo tool, not live weather
@tool
def get_weather(city: str) -> str:
    """Return demo weather data for a city."""
    return (
        f"Demo weather response for {city}: "
        "temperature 27°C (mock data, not live weather)."
    )


weather_agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=(
        "You are a weather assistant. Use the weather tool "
        "and clearly identify its output as demo data."
    ),
)


# Classify the user's question
def check_question_category(state: FlowState) -> dict:
    classifier = llm.with_structured_output(QuestionCategory)

    result = classifier.invoke(
        "Classify this question. Use 'coding' for programming, "
        "'weather' for weather, and 'google_search' for other "
        "general information. If unsure, use 'google_search'.\n"
        f"Question: {state['question']}"
    )

    return {"category": result.category}


# Router
def route(state: FlowState) -> str:
    category = state.get("category", "google_search")

    if category not in {"coding", "weather", "google_search"}:
        return "google_search"

    return category


# Coding node
def coding_node(state: FlowState) -> dict:
    response = llm.invoke(
        f"You are a helpful coding expert.\n"
        f"Question: {state['question']}"
    )

    return {"answer": str(response.content[0]["text"])}


# Google Search node
def google_node(state: FlowState) -> dict:
    result = google_agent.invoke(
        {"messages": [{"role": "user", "content": state["question"]}]}
    )

    return {"answer": str(result["messages"][-1].content[0]["text"])}


# Weather node
def weather_node(state: FlowState) -> dict:
    result = weather_agent.invoke(
        {"messages": [{"role": "user", "content": state["question"]}]}
    )

    return {"answer": str(result["messages"][-1].content[0]["text"])}


# Build the LangGraph workflow
builder = StateGraph(FlowState)

builder.add_node("classify", check_question_category)
builder.add_node("coding", coding_node)
builder.add_node("google_search", google_node)
builder.add_node("weather", weather_node)

builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    route,
    {
        "coding": "coding",
        "google_search": "google_search",
        "weather": "weather",
    },
)

builder.add_edge("coding", END)
builder.add_edge("google_search", END)
builder.add_edge("weather", END)

graph = builder.compile()


if __name__ == "__main__":
    question = input("Enter your question: ")

    response = graph.invoke(
        {
            "question": question,
            "category": "google_search",
            "answer": "",
        }
    )

    print("\nCategory:", response["category"])
    print("Answer:", response["answer"])