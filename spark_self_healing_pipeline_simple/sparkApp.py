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
    .appName(f"SELECT_TABLE_{today}") \
    .getOrCreate()

spark.sql(f"""
    SELECT customer_id, category, value1, value2
    FROM {targetTable}
    """).show()

print("SELECT TABLE COMPLETED.")

spark.stop()
