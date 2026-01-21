from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
import sys
import random
import numpy as np
from datetime import date

# Setup
targetTable = sys.argv[1]
sourceTable = sys.argv[2]

# Get the current date
today = date.today()

spark = SparkSession.builder \
    .appName(f"IcebergDynamicMultiKeySkew_{today}") \
    .getOrCreate()

spark.sql(f"""
    MERGE INTO {targetTable} AS target
    USING {sourceTable} AS source
    ON target.id = source.id
    WHEN MATCHED AND source.event_ts > target.event_ts THEN
      UPDATE SET *
    WHEN NOT MATCHED THEN
      INSERT *
""")

print("Iceberg MERGE INTO operation completed.")

spark.stop()
