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

import os
import time
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# -------------------------
# 1️⃣ Environment variables
# -------------------------
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
EMBEDDING_ENDPOINT_BASE_URL = os.environ["EMBEDDING_ENDPOINT_BASE_URL"]
EMBEDDING_CDP_TOKEN = os.environ["EMBEDDING_CDP_TOKEN"]

# -------------------------
# 2️⃣ Paths
# -------------------------
CHROMA_DIR = os.path.abspath("/home/cdsw/chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

MAPPING_DOC_PATH = os.path.abspath(
    "/home/cdsw/agentic_spark_migration_tool/migration.txt"
)

COLLECTION_NAME = "spark_submit_cde_mappings"

# -------------------------
# 3️⃣ Initialize Chroma client
# -------------------------
client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# -------------------------
# 4️⃣ Initialize embedding client
# -------------------------
llmClient = OpenAI(
    base_url=EMBEDDING_ENDPOINT_BASE_URL,
    api_key=EMBEDDING_CDP_TOKEN,
)

def get_passage_embedding(text: str):
    return llmClient.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL_ID,
        extra_body={"input_type": "passage"},
    ).data[0].embedding

# -------------------------
# 5️⃣ Chunking helper
# -------------------------
def chunk_text(text: str, max_len: int = 400):
    lines = text.split("\n")
    chunks, current = [], []

    for line in lines:
        if not line.strip():
            continue  # skip blank lines

        current.append(line)

        if sum(len(l) for l in current) >= max_len:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []

    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# -------------------------
# 6️⃣ Load mapping document
# -------------------------
def load_mapping_document(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapping file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# 7️⃣ Ingest logic
# -------------------------
def ingest_mapping_document():

    raw_text = load_mapping_document(MAPPING_DOC_PATH)
    chunks = chunk_text(raw_text)

    print(f"📄 Loaded mapping document")
    print(f"🔹 Total chunks: {len(chunks)}")

    for idx, chunk in enumerate(chunks):
        collection.add(
            ids=[f"spark_submit_mapping_001_{idx}"],
            documents=[chunk],
            metadatas=[{
                "document_id": "example_001",
                "type": "spark_submit_to_cde_mapping",
                "framework": "cdepy",
                "language": "pyspark",
                "chunk_index": idx
            }],
            embeddings=[get_passage_embedding(chunk)]
        )

        time.sleep(0.2)

    print("✅ Mapping document ingestion complete")

# -------------------------
# 8️⃣ Run ingestion
# -------------------------
if __name__ == "__main__":
    ingest_mapping_document()

    doc_count = len(collection.get()["documents"])
    print(f"📊 Collection '{COLLECTION_NAME}' document count: {doc_count}")
