from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
import sys
import random
import numpy as np
from dbldatagen import DataGenerator
from datetime import date

# Setup
targetTable = sys.argv[1]
sourceTable = sys.argv[2]

# Get the current date
today = date.today()

spark = SparkSession.builder \
    .appName(f"Datagen_{today}") \
    .getOrCreate()

spark.sql(f"DROP TABLE IF EXISTS {sourceTable} PURGE")
spark.sql(f"DROP TABLE IF EXISTS {targetTable} PURGE")

# Fixed row count for dev/testing
row_count = int(np.random.normal(loc=50_000_000, scale=50_000_000))
row_count = max(row_count, 10_000_000)  # Ensure minimum size
print(f"Generating {row_count:,} rows")

# Skew and overlap config
def create_skew_ids(num_keys=10):
    ids = random.sample(range(10_000_000, 5_000_000_000), num_keys)
    weights = np.random.dirichlet(np.ones(num_keys), size=1)[0]
    return ids, weights

# Generate skewed id set
skewed_ids, skew_weights = create_skew_ids(num_keys=10)

# Choose overlap fraction
overlap_fraction = random.uniform(0.1, 0.9)
overlap_count = int(len(skewed_ids) * overlap_fraction)
overlap_ids = skewed_ids[:overlap_count]
new_ids = [i + 10_000_000_000 for i in skewed_ids[overlap_count:]]
df2_ids = overlap_ids + new_ids
df2_weights = np.random.dirichlet(np.ones(len(df2_ids)), size=1)[0]

print(f"Overlap fraction: {overlap_fraction:.2%}")
print(f"df1 will use {len(skewed_ids)} skewed IDs")
print(f"df2 will reuse {len(overlap_ids)} (overlapping) and {len(new_ids)} (new) IDs")

num_partitions = max(row_count // 1_000_000, 4)

# ---- Create df2 ----
df2_spec = (
    DataGenerator(spark, name="df2_gen", rows=row_count, partitions=num_partitions, seedColumnName="_seed_id")
    .withColumn("event_id", "long", values=df2_ids, weights=df2_weights)
    .withColumn("category", "string", values=["A", "B", "C", "D", "E"], random=True)
    .withColumn("value1", "double", minValue=0, maxValue=1000, random=True)
    .withColumn("value2", "double", minValue=0, maxValue=100, random=True)
    .withColumn("value3", "double", minValue=0, maxValue=1000, random=True)
    .withColumn("value4", "double", minValue=0, maxValue=100, random=True)
    .withColumn("value5", "double", minValue=0, maxValue=1000, random=True)
    .withColumn("value6", "double", minValue=0, maxValue=100, random=True)
    .withColumn("value7", "double", minValue=0, maxValue=1000, random=True)
    .withColumn("value8", "double", minValue=0, maxValue=100, random=True)
    .withColumn("event_ts", "timestamp", begin="2020-01-01 01:00:00", interval="1 day", random=True)
)

df2 = df2_spec.build()
df2 = df2.drop("_seed_id")
# ---- Create df1 (only if table doesn't exist) ----
table_exists = spark._jsparkSession.catalog().tableExists(targetTable)

if not table_exists:
    print(f"Creating target table {targetTable}")

    df1_spec = (
        DataGenerator(spark, name="df1_gen", rows=row_count, partitions=num_partitions, seedColumnName="_seed_id")
        .withColumn("id", "long", values=skewed_ids, weights=skew_weights)
        .withColumn("category", "string", values=["A", "B", "C", "D", "E"], random=True)
        .withColumn("value1", "double", minValue=0, maxValue=1000, random=True)
        .withColumn("value2", "double", minValue=0, maxValue=100, random=True)
        .withColumn("value3", "double", minValue=0, maxValue=1000, random=True)
        .withColumn("value4", "double", minValue=0, maxValue=100, random=True)
        .withColumn("value5", "double", minValue=0, maxValue=1000, random=True)
        .withColumn("value6", "double", minValue=0, maxValue=100, random=True)
        .withColumn("value7", "double", minValue=0, maxValue=1000, random=True)
        .withColumn("value8", "double", minValue=0, maxValue=100, random=True)
        .withColumn("event_ts", "timestamp", begin="2020-01-01 01:00:00", interval="1 day", random=True)
    )

    df1 = df1_spec.build()
    df1 = df1.drop("_seed_id")
    df1.writeTo(targetTable).using("iceberg").create()
else:
    print(f"Target table {targetTable} exists. Skipping creation.")

# ---- Write staging table and merge ----
#spark.sql(f"DROP TABLE IF EXISTS {sourceTable} PURGE")
df2.writeTo(sourceTable).using("iceberg").create()

spark.stop()
