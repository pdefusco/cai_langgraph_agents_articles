import os
import gradio as gr
from langchain_openai import ChatOpenAI
import httpx

ON_PREM_AGENT_URL = os.getenv("ON_PREM_AGENT_URL")
ON_PREM_AGENT_ACCESS_KEY = os.getenv("ON_PREM_AGENT_ACCESS_KEY")
ON_PREM_AGENT_API_KEY = os.getenv("ON_PREM_AGENT_API_KEY")

http_client = httpx.Client(verify=False)

# LLM configuration
CLOUD_LLM = ChatOpenAI(
    model=os.getenv("CLOUD_MODEL_ID"),
    api_key=os.getenv("CLOUD_MODEL_KEY"),
    base_url=os.getenv("CLOUD_MODEL_ENDPOINT"),
    http_client=http_client
)

# === Minimal "guardrail node" ===
def guardrail_node(question: str, raw_result: str) -> str:
    """
    Very simple guardrail: summarize result for the user.
    """
    prompt = f"""
You are a safe guardrail agent.

Question: {question}

SQL Result: {raw_result}

Summarize this clearly for the end user without sensitive info.
"""
    response = CLOUD_LLM.invoke(prompt)
    return response.content.strip()

# === Gradio UI ===
def ask(question: str) -> str:
    # Simulate raw_result coming from cloud/on-prem agent
    # For now we just hardcode or call your previous function
    raw_result = [{"count(1)": 10000}]  # replace with actual backend call if desired

    # Call guardrail node directly
    answer = guardrail_node(question, raw_result)
    return answer

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Answer"),
    title="Minimal Guardrail Demo",
)

demo.launch(
    share=False,
    show_error=True,
    server_name="127.0.0.1",
    server_port=int(os.getenv("CDSW_APP_PORT")),
)
