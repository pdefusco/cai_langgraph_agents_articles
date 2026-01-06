#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

# multi_agent_langgraph_demo.py
import requests
import chromadb
from bs4 import BeautifulSoup
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
import gradio as gr
import time
import os

# -------------------------
# 1️⃣ Initialize clients
# -------------------------
MODEL_ID = os.environ["MODEL_ID"]
ENDPOINT_BASE_URL = os.environ["ENDPOINT_BASE_URL"]
CDP_TOKEN = os.environ["CDP_TOKEN"]

#openai_client = OpenAI()  # Ensure OPENAI_API_KEY is set
#embedding_model = OpenAIEmbeddings(
#    model=MODEL_ID,
#    base_url=ENDPOINT_BASE_URL,
#    api_key=CDP_TOKEN
#)

client = chromadb.PersistentClient()

# -------------------------
# 2️⃣ Scrape helper
# -------------------------
def fetch_text(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.find_all(["p", "li", "h1","h2","h3","pre"])
    return "\n".join([p.get_text(separator=" ", strip=True) for p in paragraphs])

def chunk_text(text, max_len=250):
    lines = text.split("\n")
    chunks, current = [], []
    for line in lines:
        current.append(line)
        if sum(len(l) for l in current) > max_len:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

import requests
from openai import OpenAI
import json
from tenacity import retry, wait_exponential, stop_after_attempt

llmClient = OpenAI(
    base_url=ENDPOINT_BASE_URL,  # EXACT value from AIS UI (ends with /v1)
    api_key=CDP_TOKEN,
)

def get_query_embedding(text: str):
    return llmClient.embeddings.create(
        input=f"query: {text}",
        model=MODEL_ID
    ).data[0].embedding

def get_passage_embedding(text: str):
    return llmClient.embeddings.create(
        input=f"passage: {text}",
        model=MODEL_ID
    ).data[0].embedding


# -------------------------
# 3️⃣ Chroma collections
# -------------------------
spark_col = client.get_or_create_collection("spark_tuning")
hadoop_col = client.get_or_create_collection("hadoop_perf")

# -------------------------
# 4️⃣ Optional ingestion
# -------------------------
def ingest_demo_data():
    # Spark docs
    spark_url = "https://spark.apache.org/docs/latest/tuning.html"
    spark_chunks = chunk_text(fetch_text(spark_url))
    for i, chunk in enumerate(spark_chunks):
        spark_col.add(
            documents=[chunk],
            metadatas=[{"source": "spark_tuning", "chunk_index": i}],
            embeddings=[get_query_embedding(chunk)]
        )
        time.sleep(0.5)

    # Hadoop docs
    hadoop_url = "https://openlogic.com/blog/how-to-improve-hadoop-performance"
    hadoop_chunks = chunk_text(fetch_text(hadoop_url))
    for j, chunk in enumerate(hadoop_chunks):
        hadoop_col.add(
            documents=[chunk],
            metadatas=[{"source": "hadoop_perf", "chunk_index": j}],
            embeddings=[get_query_embedding(chunk)]
        )
        time.sleep(0.5)

# =============================
# LangGraph State
# =============================
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    # This keeps track of the conversation
    messages: Annotated[List[BaseMessage], "add_messages"]
    # Your custom keys
    query: str
    spark_results: List[dict]
    hadoop_results: List[dict]
    final_answer: str

# -------------------------
# 5️⃣ LangGraph nodes
# -------------------------

def spark_retrieval(state: GraphState):
    query = state["query"]
    emb = get_passage_embedding(query)
    results = spark_col.query(query_embeddings=[emb], n_results=3)
    state["spark_results"] = [
        {"document": d, "metadata": m} for d, m in zip(results["documents"][0], results["metadatas"][0])
    ]
    return {
    "spark_results": state["spark_results"]
}


def hadoop_retrieval(state: GraphState):
    query = state["query"]
    emb = get_passage_embedding(query)
    results = hadoop_col.query(query_embeddings=[emb], n_results=3)
    state["hadoop_results"] = [
        {"document": d, "metadata": m} for d, m in zip(results["documents"][0], results["metadatas"][0])
    ]
    return {
    "hadoop_results": state["hadoop_results"]
}


def synthesis_node(state: GraphState):
    spark_docs = [r["document"] for r in state.get("spark_results", [])]
    hadoop_docs = [r["document"] for r in state.get("hadoop_results", [])]

    summary = "📄 **Summary of Big Data Solutions:**\n"
    summary += "\n**Spark Optimization:**\n" + "\n".join(f"- {d}" for d in spark_docs) + "\n"
    summary += "\n**Hadoop Optimization:**\n" + "\n".join(f"- {d}" for d in hadoop_docs) + "\n\n"

    # Detect overlapping keywords as simple conflict detection
    spark_words = set(" ".join(spark_docs).lower().split())
    hadoop_words = set(" ".join(hadoop_docs).lower().split())
    overlap = spark_words.intersection(hadoop_words)
    if overlap:
        summary += f"⚠️ Overlapping terms: {', '.join(list(overlap)[:10])}\n"

    summary += "\n💡 Recommendation: Consider both application-level and storage-level optimizations, and ensure caching/parallelism strategies do not conflict."
    state["final_answer"] = summary
    return {"state": state}

# -------------------------
# 6️⃣ Build LangGraph
# -------------------------
graph = StateGraph(GraphState)

graph.add_node("spark_agent", spark_retrieval)
graph.add_node("hadoop_agent", hadoop_retrieval)
graph.add_node("synthesis_agent", synthesis_node)

# Flow edges
graph.add_edge(START, "spark_agent")
graph.add_edge(START, "hadoop_agent")
graph.add_edge("spark_agent", "synthesis_agent")
graph.add_edge("hadoop_agent", "synthesis_agent")
graph.add_edge("synthesis_agent", END)

compiled = graph.compile()

# -------------------------
# 7️⃣ Runner function
# -------------------------
def run_langgraph(query: str) -> str:
    # 🛡️ Safety check to prevent NoneType errors
    if not query:
        return "Please enter a question."

    inputs = {
        "query": query,
        "messages": [{"role": "user", "content": query}]
    }

    # Ensure you are using the logic from the previous fix for GraphState
    out = compiled.invoke(inputs)

    # Return the string directly for the Gradio Markdown/Textbox component
    return out.get("final_answer", "No answer generated.")


# -------------------------
# 8️⃣ Gradio UI
# -------------------------
sample_questions = [
    "Our Spark jobs are running slowly, how can we optimize?",
    "Hadoop MapReduce tasks are timing out under high load",
    "How can we improve cluster-wide job throughput?",
    "What caching strategies are recommended for big data pipelines?"
]

with gr.Blocks() as demo:
    gr.Markdown("## Big Data Multi-Agent QA")
    input_box = gr.Textbox(label="Enter your question here")
    output_box = gr.Markdown()
    submit_btn = gr.Button("Submit")

    # CRITICAL: Ensure inputs=[input_box] only matches the 1 argument in run_langgraph
    submit_btn.click(
        fn=run_langgraph,
        inputs=[input_box],
        outputs=[output_box]
    )

    # Also handle 'Enter' key press
    input_box.submit(
        fn=run_langgraph,
        inputs=[input_box],
        outputs=[output_box]
    )


demo.queue(default_concurrency_limit=16).launch(share=False,
            show_error=True,
            server_name='127.0.0.1',
            server_port=int(os.getenv('CDSW_APP_PORT')))
