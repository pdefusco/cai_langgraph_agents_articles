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

CLOUD_LLM = ChatOpenAI(
    model=os.getenv("CLOUD_MODEL_ID"),
    api_key=os.getenv("CLOUD_MODEL_KEY"),
    base_url=os.getenv("CLOUD_MODEL_ENDPOINT"),
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

# =========================================================
# A2A Client Node (Cloud → On-Prem)
# =========================================================

import requests
import os

CLOUD_AGENT_URL = os.getenv("CLOUD_AGENT_URL") + "invoke"
CLOUD_AGENT_API_KEY = os.getenv("CLOUD_AGENT_API_KEY")  # optional

def call_cloud_agent(question: str) -> dict:
    print(">>> call_cloud_agent start")
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
    print(">>> call_cloud_agent got response:", response.status_code)
    response.raise_for_status()
    result = response.json()
    print(">>> call_cloud_agent json:", result)
    return result


def cloud_node(state: State) -> State:
    result = call_cloud_agent(state["question"])
    sql = result["sql"].strip().strip("`")
    return {
        **state,
        "sql": result["sql"],
        "raw_result": result["result"],
    }

# =========================================================
# Guardrail Agent (Cloud)
# =========================================================

def guardrail_node(state: State) -> State:
    prompt = f"Summarize this SQL query result for an end user:\n\nResult: {state['raw_result']}"
    print(">>> calling CLOUD_LLM (non-streaming)")
    response = CLOUD_LLM.invoke(prompt)
    answer_text = str(response.content).strip()
    print(">>> CLOUD_LLM done. answer:", answer_text)

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
'''    result = langgraph_app.invoke({"question": question}, stream=False)
    print("ask() full result:", result, type(result))
    print(json.dumps(result, indent=2))'''
    result = [{"count(1)": 10000}]
    # Extract answer from the guardrail node
    guardrail_result = result.get("guardrail", {})
    answer_text = guardrail_result.get("answer", "No result returned")

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
