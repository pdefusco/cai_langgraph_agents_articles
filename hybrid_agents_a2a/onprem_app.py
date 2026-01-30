import os, json
import requests
import gradio as gr
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import httpx

http_client = httpx.Client(verify=False)

# =========================================================
# Configuration
# =========================================================

ON_PREM_AGENT_URL = os.getenv("ON_PREM_AGENT_URL")
ON_PREM_AGENT_ACCESS_KEY = os.getenv("ON_PREM_AGENT_ACCESS_KEY")
ON_PREM_AGENT_API_KEY = os.getenv("ON_PREM_AGENT_API_KEY")

ONPREM_LLM = ChatOpenAI(
    model=os.getenv("ONPREM_MODEL_ID"),
    api_key=os.getenv("ONPREM_MODEL_KEY"),
    base_url=os.getenv("ONPREM_MODEL_ENDPOINT"),
    http_client=http_client
)

# =========================================================
# LangGraph State
# =========================================================

class State(TypedDict):
    question: str
    sql: str
    raw_result: str
    answer: str
    table: str

# =========================================================
# A2A Client Node (Cloud → On-Prem)
# =========================================================

import requests
import os

ON_PREM_AGENT_URL = os.getenv("ON_PREM_AGENT_URL") + "invoke"
ON_PREM_AGENT_API_KEY = os.getenv("ON_PREM_AGENT_API_KEY")  # optional

def call_cloud_agent(question: str) -> dict:
    print(">>> call_cloud_agent start")

    payload = {
        "contract": {
            "requested_tables": ["TableTest"]
        },
        "request": {
            "question": question
        }
    }

    headers = {
        "Content-Type": "application/json",
    }

    if ON_PREM_AGENT_API_KEY:
        headers["Authorization"] = f"Bearer {ON_PREM_AGENT_API_KEY}"

    response = requests.post(
        ON_PREM_AGENT_URL,
        json=payload,
        headers=headers,
        timeout=30,
    )

    print(">>> call_cloud_agent got response:", response.status_code)
    response.raise_for_status()
    result = response.json()
    print(">>> call_cloud_agent json:", result)
    return result

def cloud_node(state: State) -> State:
    result = call_cloud_agent(state["question"])
    return {
        **state,
        "sql": result["sql"],
        "raw_result": result["result"],
        "table": result["table"],
    }

# =========================================================
# Guardrail Agent (Cloud)
# =========================================================

def detect_table_mentions(question: str) -> set[str]:
    KNOWN_TABLES = {"TableTest", "Customers", "Payroll"}
    return {t for t in KNOWN_TABLES if t.lower() in question.lower()}

def guardrail_node(state: State) -> State:
    prompt = f"Summarize this SQL query result for an end user:\n\nResult: {state['raw_result']}"
    print(">>> calling ONPREM_LLM (non-streaming)")
    response = ONPREM_LLM.invoke(prompt)
    answer_text = str(response.content).strip()
    print(">>> ONPREM_LLM done. answer:", answer_text)

    return {
        **state,
        "answer": answer_text
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

'''def ask(question: str) -> str:
    result = langgraph_app.invoke({"question": question}, stream=False)
    print("ask() full result:", result)

    # If LangGraph returns node outputs nested:
    if "guardrail" in result and "answer" in result["guardrail"]:
        return result["guardrail"]["answer"]

    return "No result returned"'''

def ask(question: str) -> str:
    # Invoke LangGraph in non-streaming mode
    mentioned = detect_table_mentions(question)

    if mentioned and mentioned != {"TableTest"}:
        return (
            "You’re currently authorized to query TableTest only. "
            f"Your question mentions: {', '.join(mentioned)}."
        )

    result = langgraph_app.invoke({"question": question}, stream=False)
    print("ask() full result:", result, type(result))
    print(json.dumps(result, indent=2))

    # Extract answer from the guardrail node
    answer_text = result.get("answer", "No result returned")
    return answer_text

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
