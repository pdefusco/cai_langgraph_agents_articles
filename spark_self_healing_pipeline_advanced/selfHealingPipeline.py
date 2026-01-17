#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#****************************************************************************

import json
import time
import threading
import os
import difflib
from typing import TypedDict

import gradio as gr

from cdepy import (
    cdeconnection,
    cdemanager,
    cdejob,
    cderesource
)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

import gradio as gr
print("Gradio runtime version:", gr.__version__)


# =========================================================
# USER CONFIGURATION
# =========================================================

JOBS_API_URL = os.environ["JOBS_API_URL"]
WORKLOAD_USER = os.environ["WORKLOAD_USER"]
WORKLOAD_PASSWORD = os.environ["WORKLOAD_PASSWORD"]

#JOB_NAME = os.environ["JOB_NAME"]
RESOURCE_NAME = os.environ["RESOURCE_NAME"]
#APPLICATION_FILE_NAME = os.environ["APPLICATION_FILE_NAME"]

LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
LLM_ENDPOINT_BASE_URL = os.environ["LLM_ENDPOINT_BASE_URL"]
LLM_CDP_TOKEN = os.environ["LLM_CDP_TOKEN"]

POLL_INTERVAL_SECONDS = 30

LAST_REMEDIATION_INFO = {
    "summary": "",
    "job_name": "",
    "resource_name": "",
}

REMEDIATED_JOBS: set[str] = set()

# =========================================================
# GLOBAL CDE OBJECTS
# =========================================================

CDE_CONNECTION = None
CDE_MANAGER = None


def init_cde():
    global CDE_CONNECTION, CDE_MANAGER
    CDE_CONNECTION = cdeconnection.CdeConnection(
        JOBS_API_URL,
        WORKLOAD_USER,
        WORKLOAD_PASSWORD,
    )
    CDE_CONNECTION.setToken()
    CDE_MANAGER = cdemanager.CdeClusterManager(CDE_CONNECTION)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(TypedDict):
    latest_run_id: str | None
    latest_run_status: str | None
    latest_job_name: str | None

    application_file_name: str | None
    spark_logs: str | None
    spark_script: str | None

    llm_analysis: str | None
    improved_script: str | None
    code_diff: str | None

    remediation_summary: str | None
    new_job_name: str | None
    new_resource_name: str | None

    retried: bool

LAST_AGENT_STATE: AgentState | None = None

# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0,
)


# =========================================================
# GRAPH NODES
# =========================================================

import os
import json

LLM_FAILING_JOBS = [
    "llm-failing-job-1",
    "llm-failing-job-2",
    "llm-failing-job-3",
    "llm-failing-job-4",
    "llm-failing-job-5",
]

def fetch_latest_run(state: dict = None) -> dict:
    if state is None:
        state = {}

    result = CDE_MANAGER.listJobRuns()
    if result == -1 or not result:
        return state

    runs = json.loads(result).get("runs", [])

    # Group latest run per job
    latest_by_job = {}
    for r in runs:
        job = r.get("job")
        if job in LLM_FAILING_JOBS:
            if job not in latest_by_job or r["started"] > latest_by_job[job]["started"]:
                latest_by_job[job] = r

    # Pick the first FAILED job that has NOT been remediated
    for job_name in LLM_FAILING_JOBS:
        if job_name in REMEDIATED_JOBS:
            continue

        run = latest_by_job.get(job_name)
        if run and run.get("status", "").upper() == "FAILED":
            state["latest_job_name"] = job_name
            state["latest_run_id"] = str(run["id"])
            state["latest_run_status"] = "FAILED"
            return state

    # Nothing left to remediate
    state["latest_run_id"] = None
    state["latest_run_status"] = None
    return state



def route_on_status(state: AgentState):
    if state["latest_run_status"] == "FAILED" and not state["retried"]:
        return "download_artifacts"
    return END


def download_artifacts(state: AgentState):
    run_id = state["latest_run_id"]
    job_name = state["latest_job_name"]

    # Logs
    logs = CDE_MANAGER.downloadJobRunLogs(run_id, "driver/stdout")
    state["spark_logs"] = logs or "No driver stdout logs available"

    print("\n========== DRIVER STDOUT LOGS ==========")
    print(state["spark_logs"])
    print("========== END DRIVER STDOUT LOGS ==========\n")

    describe_dict = json.loads(CDE_MANAGER.describeJob(job_name))
    application_file_name = describe_dict["spark"]["file"]

    state["application_file_name"] = application_file_name

    # Download script
    script = CDE_MANAGER.downloadFileFromResource(
        RESOURCE_NAME,
        application_file_name,
    )

    state["spark_script"] = script or ""
    return state

import re
import difflib

def llm_analyze_and_fix(state: AgentState):
    """
    Analyze a failing Spark script, produce a fixed version using the LLM,
    and clean the output to ensure only valid Python code is deployed.
    """
    # === 1️⃣ Prepare prompt for LLM ===
    prompt = [
        SystemMessage(
            content=(
                "You are a senior Spark engineer.\n"
                "1. Identify the root cause of failure\n"
                "2. Explain why it happened\n"
                "3. Mention alternatives if relevant\n"
                "4. Produce a **complete, ready-to-run Spark Python script**.\n\n"
                "Respond EXACTLY in this format:\n"
                "=== ANALYSIS ===\n"
                "<analysis>\n\n"
                "=== FIXED SCRIPT ===\n"
                "<full python script, no placeholders, no backticks>"
            )
        ),
        HumanMessage(
            content=(
                f"SPARK SCRIPT:\n{state['spark_script']}\n\n"
                f"DRIVER STDOUT LOGS:\n{state['spark_logs']}"
            )
        ),
    ]

    # === 2️⃣ Invoke LLM ===
    response = llm.invoke(prompt)
    text = response.content

    # === 3️⃣ Extract analysis and fixed script ===
    if "=== FIXED SCRIPT ===" in text:
        analysis, raw_fixed_script = text.split("=== FIXED SCRIPT ===", 1)
    else:
        analysis = "No analysis returned."
        raw_fixed_script = state["spark_script"]  # fallback

    # === 4️⃣ Clean fixed script ===
    def extract_fixed_script(script_text: str) -> str:
        lines = script_text.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip markdown, headings, or explanatory text
            if line.startswith("```"):
                continue
            if line.startswith("#") and not line.startswith("# "):  # keep proper code comments
                continue
            if any(line.lower().startswith(prefix) for prefix in ("note:", "explanation:", "troubleshoot", "**")):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    fixed_script = extract_fixed_script(raw_fixed_script)

    # === 5️⃣ Optional: Validate Python syntax ===
    try:
        compile(fixed_script, "<string>", "exec")
    except Exception as e:
        print(f"[LLM FIX ERROR] Python compilation failed: {e}")
        # fallback to original script if compilation fails
        fixed_script = state["spark_script"]

    # === 6️⃣ Generate diff for visibility ===
    diff = difflib.unified_diff(
        state["spark_script"].splitlines(),
        fixed_script.splitlines(),
        fromfile="original.py",
        tofile="fixed.py",
        lineterm="",
    )

    # === 7️⃣ Update AgentState ===
    state["llm_analysis"] = analysis.replace("=== ANALYSIS ===", "").strip()
    state["improved_script"] = fixed_script
    state["code_diff"] = "\n".join(diff)
    state["retried"] = True

    return state


import tempfile
import os

def deploy_and_run_fixed_job(state: AgentState):

    base_job_name = state["latest_job_name"]
    base_job_name_safe = base_job_name.replace("-", "_")

    new_resource = f"{RESOURCE_NAME}-fixed"
    new_job_name = f"{base_job_name}-fixed"

    CDE_RESOURCE = cderesource.CdeFilesResource(new_resource)
    cdeFilesResourceDefinition = CDE_RESOURCE.createResourceDefinition()

    CDE_MANAGER.createResource(cdeFilesResourceDefinition)

    # ✅ WRITE SCRIPT TO LOCAL FILE FIRST
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False
    ) as f:
        f.write(state["improved_script"])
        local_path = f.name

    local_dir = os.path.dirname(local_path)
    local_file = os.path.basename(local_path)

    # ✅ NOW upload using cdepy expectations
    CDE_MANAGER.uploadFileToResource(
        new_resource,
        local_dir,
        local_file,
    )

    spark_job = cdejob.CdeSparkJob(CDE_CONNECTION)
    job_def = spark_job.createJobDefinition(
        new_job_name,
        new_resource,
        local_file,
        executorMemory="2g",
        executorCores=2,
        pythonEnvResourceName="datagen-env",
            args=[f"spark_catalog.default.target_table_{base_job_name_safe}",
                    f"spark_catalog.default.source_table_{base_job_name_safe}"]
        )
    )

    CDE_MANAGER.createJob(job_def)
    CDE_MANAGER.runJob(new_job_name)

    REMEDIATED_JOBS.add(base_job_name)

    summary = (
        f"New job created: {new_job_name}\n"
        f"New resource created: {new_resource}\n"
        f"Application File: {state.get('application_file_name', 'N/A')}"
        f"Job submitted successfully."
    )

    state["remediation_summary"] = summary
    state["new_job_name"] = new_job_name
    state["new_resource_name"] = new_resource

    # ✅ GLOBAL CACHE FOR UI
    global LAST_REMEDIATION_INFO
    LAST_REMEDIATION_INFO = {
        "summary": summary,
        "job_name": new_job_name,
        "resource_name": new_resource,
    }

    return state


# =========================================================
# LANGGRAPH
# =========================================================

graph = StateGraph(AgentState)

graph.add_node("fetch_latest_run", fetch_latest_run)
graph.add_node("download_artifacts", download_artifacts)
graph.add_node("llm_analyze_and_fix", llm_analyze_and_fix)
graph.add_node("deploy_and_run_fixed_job", deploy_and_run_fixed_job)

graph.set_entry_point("fetch_latest_run")

graph.add_conditional_edges(
    "fetch_latest_run",
    route_on_status,
    {"download_artifacts": "download_artifacts", END: END},
)

graph.add_edge("download_artifacts", "llm_analyze_and_fix")
graph.add_edge("llm_analyze_and_fix", "deploy_and_run_fixed_job")
graph.add_edge("deploy_and_run_fixed_job", END)

app = graph.compile()


# =========================================================
# AGENT LOOP
# =========================================================

def run_monitor():
    global LAST_AGENT_STATE

    result = app.invoke(
        {
            "latest_run_id": None,
            "latest_run_status": None,
            "latest_job_name": None,
            "application_file_name": None,
            "spark_logs": None,
            "spark_script": None,
            "llm_analysis": None,
            "improved_script": None,
            "code_diff": None,
            "new_job_name": None,
            "new_resource_name": None,
            "retried": False,
        }
    )

    LAST_AGENT_STATE = result



def agent_loop():
    while True:
        run_monitor()
        time.sleep(POLL_INTERVAL_SECONDS)


# =========================================================
# UI
# =========================================================

def ui_refresh():
    global LAST_AGENT_STATE

    if not LAST_AGENT_STATE:
        return (
            "Waiting for agent...",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )

    state = LAST_AGENT_STATE

    status_text = (
        f"Latest Failing Job: {state.get('latest_job_name', 'N/A')}\n"
        f"Run ID: {state.get('latest_run_id', 'N/A')}\n"
        f"Status: {state.get('latest_run_status', 'UNKNOWN')}\n"
        f"Application File: {state.get('application_file_name', 'N/A')}"
    )

    remediation_summary_text = state.get(
        "remediation_summary", "No remediation info yet."
    )

    updated_job_text = (
        f"Job Name: {state.get('new_job_name', 'N/A')}\n"
        f"Resource Name: {state.get('new_resource_name', 'N/A')}"
    )

    return (
        status_text,
        remediation_summary_text,
        updated_job_text,
        state.get("spark_script", ""),
        state.get("spark_logs", ""),
        state.get("llm_analysis", ""),
        state.get("improved_script", ""),
        state.get("code_diff", ""),
    )


# =========================================================
# STARTUP
# =========================================================

init_cde()

def start_agent():
    threading.Thread(target=agent_loop, daemon=True).start()

css = """
/* Page title at the top */
.page-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 16px;
    text-align: center;
}

/* Titles inside status/remediation boxes */
.status-box-title {
    font-weight: bold;
    font-size: 16px;
    margin-bottom: 4px;
    color: #333;
}

/* Boxes themselves */
.status-box {
    border: 1px solid #ccc;
    padding: 8px;
    border-radius: 5px;
    background-color: #f9f9f9;
}

/* Scrollable code for spark scripts */
.scrollable-code .gr-code {
    max-height: 300px;
    overflow-y: auto;
}
"""

with gr.Blocks(title="CDE Spark Job Monitor & Auto Remediator", css=css) as demo:

    # Page title
    gr.Markdown("<div class='page-title'>CDE Spark Job Monitor & Auto Remediator</div>")

    # Top row: show latest failing job, remediation summary, updated job info
    with gr.Row():
        latest_job_box = gr.Textbox(
            label="Latest Failing Job Being Processed",
            lines=2,
            interactive=False,
        )
        remediation_summary_box = gr.Textbox(
            label="Remediation Summary",
            lines=5,
            interactive=False
        )
        updated_job_box = gr.Textbox(
            label="Updated Job (Remediated) Information",
            lines=5,
            interactive=False
        )

    # Second row: code, logs, analysis
    with gr.Row():
        script_box = gr.Code(
            label="Spark Script",
            language="python",
            show_label=True,
            interactive=False,
            elem_classes=["scrollable-code"]
        )
        fixed_script_box = gr.Code(
            label="Improved Spark Script",
            language="python",
            show_label=True,
            interactive=False,
            elem_classes=["scrollable-code"]
        )

    # Third row: logs, LLM analysis, code diff
    with gr.Row():
        logs_box = gr.Textbox(
            label="Driver Stdout Logs",
            lines=15,
            max_lines=15
        )
        analysis_box = gr.Textbox(
            label="LLM Analysis (Root Cause & Explanation)",
            lines=10,
            max_lines=10
        )
        diff_box = gr.Textbox(
            label="Spark Code Diff (Original vs Fixed)",
            lines=20,
            max_lines=20
        )

    # Timer to refresh UI every 10 seconds
    timer = gr.Timer(value=10, active=True)  # refresh every 2 minutes
    timer.tick(
        fn=ui_refresh,
        inputs=[],
        outputs=[
            latest_job_box,
            remediation_summary_box,
            updated_job_box,
            script_box,
            logs_box,
            analysis_box,
            fixed_script_box,
            diff_box,
        ]
    )

    demo.load(fn=start_agent)


if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT")),
    )
