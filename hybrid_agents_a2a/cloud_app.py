import os
import requests
import gradio as gr
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI

# =========================================================
# Configuration
# =========================================================

ON_PREM_AGENT_URL = "https://onprem-agent.company.com/invoke"

CLOUD_LLM = ChatOpenAI(
    model="nemotron-cloud",
    api_key=os.getenv("CLOUD_NEMOTRON_KEY"),
    base_url=os.getenv("CLOUD_NEMOTRON_ENDPOINT"),
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

def call_on_prem_agent(question: str) -> dict:
    response = requests.post(
        ON_PREM_AGENT_URL,
        json={"question": question},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()

def on_prem_node(state: State) -> State:
    result = call_on_prem_agent(state["question"])
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

graph.add_node("on_prem", on_prem_node)
graph.add_node("guardrail", guardrail_node)

graph.set_entry_point("on_prem")
graph.add_edge("on_prem", "guardrail")
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
