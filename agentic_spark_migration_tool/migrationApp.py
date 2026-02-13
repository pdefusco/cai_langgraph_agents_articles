import os
import shutil
from typing import Dict, Any, List
import gradio as gr
from pydantic import BaseModel, ValidationError
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import chromadb
from chromadb.config import Settings
from cdepy import (
    cdeconnection,
    cdemanager,
    cdejob,
    cderesource
)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SCRIPTS_DIR = "scripts"
os.makedirs(SCRIPTS_DIR, exist_ok=True)

JOBS_API_URL = os.environ["JOBS_API_URL"]
WORKLOAD_USER = os.environ["WORKLOAD_USER"]
WORKLOAD_PASSWORD = os.environ["WORKLOAD_PASSWORD"]

JOB_NAME = os.environ["JOB_NAME"]
RESOURCE_NAME = os.environ["RESOURCE_NAME"]
#APPLICATION_FILE_NAME = os.environ["APPLICATION_FILE_NAME"]

LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
LLM_ENDPOINT_BASE_URL = os.environ["LLM_ENDPOINT_BASE_URL"]
LLM_CDP_TOKEN = os.environ["LLM_CDP_TOKEN"]

CHROMA_DIR = "/home/cdsw/chroma"
COLLECTION_NAME = "spark_submit_cde_mappings"

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

init_cde()

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    base_url=LLM_ENDPOINT_BASE_URL,
    api_key=LLM_CDP_TOKEN,
    temperature=0.2,
)

# -------------------------------------------------------------------
# Structured LLM for CDE Specs
# -------------------------------------------------------------------

structured_cde_llm = llm.with_structured_output(GeneratedCdeSpecs)

# -------------------------------------------------------------------
# Chroma (existing collection)
# -------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

spark_submit_collection = chroma_client.get_collection(
    COLLECTION_NAME
)

# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------

class ParsedSparkSubmit(BaseModel):
    executor_memory: str | None = None
    executor_cores: int | None = None
    num_executors: int | None = None
    spark_conf: Dict[str, str] = {}
    args: List[str] = []


class CdeFilesResourceSpec(BaseModel):
    resource_name: str
    local_files: List[str]


class CdeSparkJobSpec(BaseModel):
    job_name: str
    resource_name: str
    application_file: str
    executorMemory: str | None = None
    executorCores: int | None = None
    numExecutors: int | None = None
    spark_conf: Dict[str, str] = {}
    args: List[str] = []


class AgentState(BaseModel):
    spark_submit: str
    script_name: str
    local_script_path: str

    job_name: str
    resource_name: str

    parsed_submit: ParsedSparkSubmit | None = None
    rag_examples: List[str] = []

    resource_spec: CdeFilesResourceSpec | None = None
    job_spec: CdeSparkJobSpec | None = None

    errors: List[str] = []

class GeneratedCdeSpecs(BaseModel):
    cde_files_resource: CdeFilesResourceSpec
    cde_spark_job: CdeSparkJobSpec


# -------------------------------------------------------------------
# LangGraph Nodes
# -------------------------------------------------------------------

def parse_spark_submit(state: AgentState) -> AgentState:
    prompt = f"""
    Extract parameters from this spark-submit command.
    Return ONLY valid JSON.
    No explanations.

    Spark submit:
    {state.spark_submit}
    """

    response = llm.invoke(prompt).content

    json_start = response.find("{")
    json_str = response[json_start:]

    state.parsed_submit = ParsedSparkSubmit.model_validate_json(json_str)

    return state


def retrieve_examples(state: AgentState) -> AgentState:
    results = spark_submit_collection.query(
        query_texts=[state.spark_submit],
        n_results=3
    )

    state.rag_examples = results["documents"][0]
    return state


def generate_cde_specs(state: AgentState) -> AgentState:

    prompt = f"""
    You are a Python coding agent tasked with creating Cloudera Data Engineering (CDE) Spark jobs
    using the cdepy library.

    Follow retrieved examples exactly.
    Map --conf flags only to spark_conf.
    Do not modify job_name or resource_name.
    Do not hallucinate values.

    job_name: {state.job_name}
    resource_name: {state.resource_name}
    application_file: {state.script_name}

    Retrieved examples:
    {state.rag_examples}

    Parsed spark-submit:
    {state.parsed_submit.model_dump()}
    """

    result: GeneratedCdeSpecs = structured_cde_llm.invoke(prompt)

    state.resource_spec = result.cde_files_resource
    state.job_spec = result.cde_spark_job

    return state


def validate_specs(state: AgentState) -> AgentState:
    errors = []

    if state.script_name not in state.resource_spec.local_files:
        errors.append("Uploaded script missing from CDE resource")

    if state.job_spec.application_file != state.script_name:
        errors.append("Application file mismatch")

    if errors:
        state.errors = errors

    return state


def execute_cde(state: AgentState) -> AgentState:

    manager = CDE_MANAGER  # use global initialized manager

    # Create resource
    resource = cderesource.CdeFilesResource(
        state.resource_spec.resource_name
    )
    resource_def = resource.createResourceDefinition()
    manager.createResource(resource_def)

    # Upload script
    manager.uploadFileToResource(
        state.resource_spec.resource_name,
        os.path.dirname(state.local_script_path),
        state.script_name
    )

    # Create job
    spark_job = cdejob.CdeSparkJob(CDE_CONNECTION)

    job_def = spark_job.createJobDefinition(
        state.job_spec.job_name,
        state.job_spec.resource_name,
        state.job_spec.application_file,
        executorMemory=state.job_spec.executorMemory,
        executorCores=state.job_spec.executorCores,
        numExecutors=state.job_spec.numExecutors
    )

    manager.createJob(job_def)

    return state


# -------------------------------------------------------------------
# LangGraph wiring
# -------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("parse_submit", parse_spark_submit)
graph.add_node("retrieve_examples", retrieve_examples)
graph.add_node("generate_specs", generate_cde_specs)
graph.add_node("validate", validate_specs)
graph.add_node("execute", execute_cde)

graph.set_entry_point("parse_submit")

graph.add_edge("parse_submit", "retrieve_examples")
graph.add_edge("retrieve_examples", "generate_specs")
graph.add_edge("generate_specs", "validate")

graph.add_conditional_edges(
    "validate",
    lambda s: "execute" if not s.errors else END
)

graph.add_edge("execute", END)

agent = graph.compile()

# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------

def run_agent(spark_submit_text, pyspark_file):

    filename = os.path.basename(pyspark_file.name)

    state = AgentState(
        spark_submit=spark_submit_text,
        script_name=filename,
        local_script_path=pyspark_file.name,
        job_name=JOB_NAME,
        resource_name=RESOURCE_NAME
    )

    final_state = agent.invoke(state)

    if final_state.errors:
        return {
            "status": "error",
            "errors": final_state.errors
        }

    return {
        "status": "success",
        "resource": final_state.resource_spec.model_dump(),
        "job": final_state.job_spec.model_dump()
    }


with gr.Blocks() as demo:
    gr.Markdown("## Create CDE Spark Job (LangGraph + RAG)")

    spark_submit = gr.Textbox(
        label="Spark Submit Command",
        lines=6
    )

    pyspark_file = gr.File(
        label="Upload PySpark Script (.py)",
        file_types=[".py"]
    )

    output = gr.JSON(label="Result")

    btn = gr.Button("Create CDE Job")
    btn.click(run_agent, [spark_submit, pyspark_file], output)


if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=int(os.getenv("CDSW_APP_PORT")),
    )
