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
import time
import os

# -------------------------
# 1️⃣ Initialize clients
# -------------------------
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
EMBEDDING_ENDPOINT_BASE_URL = os.environ["EMBEDDING_ENDPOINT_BASE_URL"]
EMBEDDING_CDP_TOKEN = os.environ["EMBEDDING_CDP_TOKEN"]

# -------------------------
# 2️⃣ Initialize Chroma client (persistent)
# -------------------------
CHROMA_DIR = "/home/cdsw/chroma"  # persist all collections here
os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.PersistentClient(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DIR
    )
)

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
    base_url=EMBEDDING_ENDPOINT_BASE_URL,  # EXACT value from AIS UI (ends with /v1)
    api_key=EMBEDDING_CDP_TOKEN,
)

def get_query_embedding(text: str):
    return llmClient.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL_ID,
        extra_body={"input_type": "query"},
    ).data[0].embedding

def get_passage_embedding(text: str):
    return llmClient.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL_ID,
        extra_body={"input_type": "passage"},
    ).data[0].embedding

# -------------------------
# 3️⃣ Chroma collections
# -------------------------
spark_col = client.get_or_create_collection("spark_tuning")
hadoop_col = client.get_or_create_collection("hadoop_perf")

# -------------------------
# 4️⃣ Chroma ingestion
# -------------------------
def ingest_demo_data():
    # Spark docs
    spark_url = "https://spark.apache.org/docs/latest/tuning.html"
    spark_chunks = chunk_text(fetch_text(spark_url))
    for i, chunk in enumerate(spark_chunks):
        spark_col.add(
            ids=[f"spark_{i}"],  # <--- unique ID for each chunk
            documents=[chunk],
            metadatas=[{"source": "spark_tuning", "chunk_index": i}],
            embeddings=[get_passage_embedding(chunk)]  # match the retrieval method
        )
        time.sleep(0.5)

    # Hadoop docs
    hadoop_url = "https://openlogic.com/blog/how-to-improve-hadoop-performance"
    hadoop_chunks = chunk_text(fetch_text(hadoop_url))
    for j, chunk in enumerate(hadoop_chunks):
        hadoop_col.add(
            ids=[f"hadoop_{j}"],  # <--- unique ID for each chunk
            documents=[chunk],
            metadatas=[{"source": "hadoop_perf", "chunk_index": j}],
            embeddings=[get_passage_embedding(chunk)]  # match retrieval
        )
        time.sleep(0.5)


ingest_demo_data()
print("Spark docs:", len(spark_col.get()["documents"]))
print("Hadoop docs:", len(hadoop_col.get()["documents"]))
