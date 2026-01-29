import os, json
import requests
import gradio as gr
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


# =========================================================
# Configuration
# =========================================================

ON_PREM_AGENT_URL = os.getenv("ON_PREM_AGENT_URL")
ON_PREM_AGENT_ACCESS_KEY = os.getenv("ON_PREM_AGENT_ACCESS_KEY")
ON_PREM_AGENT_API_KEY = os.getenv("ON_PREM_AGENT_API_KEY")

CLOUD_LLM = ChatOpenAI(
    model=os.getenv("CLOUD_MODEL_ID"),
    api_key=os.getenv("CLOUD_MODEL_KEY"),
    base_url=os.getenv("CLOUD_MODEL_ENDPOINT"),
)

# =========================================================
# LangGraph State
# =========================================================

class State(TypedDict):
    question: str
    sql: str
    raw_result: str
    answer: str

# =========================================================
# A2A Client Node (Cloud → On-Prem)
# =========================================================

import requests
import os

CLOUD_AGENT_URL = os.getenv("CLOUD_AGENT_URL") + "/invoke"
CLOUD_AGENT_API_KEY = os.getenv("CLOUD_AGENT_API_KEY")  # optional

def call_cloud_agent(question: str) -> dict:
    payload = {
        "request": {
            "question": question
        }
    }

    headers = {
        "Content-Type": "application/json",
    }

    # Only include Authorization if you actually enforce it
    if CLOUD_AGENT_API_KEY:
        headers["Authorization"] = f"Bearer {CLOUD_AGENT_API_KEY}"

    response = requests.post(
        CLOUD_AGENT_URL,
        json=payload,      # <-- IMPORTANT
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()



def cloud_node(state: State) -> State:
    result = call_cloud_agent(state["question"])
    return {
        **state,
        "sql": result["sql"],
        "raw_result": result["result"],
    }

# =========================================================
# Guardrail Agent (Cloud)
# =========================================================

def guardrail_node(state: State) -> State:
    prompt = f"""
You are a guardrail agent.

- Ensure the answer is safe
- Ensure no sensitive data is exposed
- Summarize clearly for an end user

SQL:
{state['sql']}

Result:
{state['raw_result']}
"""

    response = CLOUD_LLM.invoke(prompt)

    return {
        **state,
        "answer": response.content
    }

# =========================================================
# LangGraph Definition
# =========================================================

graph = StateGraph(State)

graph.add_node("cloud", cloud_node)
graph.add_node("guardrail", guardrail_node)

graph.set_entry_point("cloud")
graph.add_edge("cloud", "guardrail")
graph.add_edge("guardrail", END)

langgraph_app = graph.compile()

# =========================================================
# Gradio UI
# =========================================================

def ask(question: str) -> str:
    result = langgraph_app.invoke({"question": question})
    return result["answer"]

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Answer"),
    title="Cloud ↔ On-Prem A2A Demo",
)

demo.launch(
    share=False,
    show_error=True,
    server_name="127.0.0.1",
    server_port=int(os.getenv("CDSW_APP_PORT")),
)
