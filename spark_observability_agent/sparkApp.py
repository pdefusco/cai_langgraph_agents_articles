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

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, col, lit

spark = (SparkSession
  .builder
  .appName("Test sparkmeasure instrumentation of Python/PySpark code")
  .config("spark.jars.packages","ch.cern.sparkmeasure:spark-measure_2.12:0.27")
  .config("spark.kerberos.access.hadoopFileSystems","s3a://go01-demo/")
  .getOrCreate() )

app_id = spark.sparkContext.applicationId


from sparkmeasure import TaskMetrics

taskmetrics = TaskMetrics(spark)
taskmetrics.runandmeasure(globals(), 'spark.sql("select count(*) from range(1000) cross join range(1000) cross join range(1000)").show()')
taskMetricsDf = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

# from_unixtime returns a string by default, so cast it to timestamp
#taskMetricsDf = taskMetricsDf.withColumn("finishTime", from_unixtime(col("finishTime")).cast("timestamp"))

from pyspark.sql.functions import col, timestamp_seconds

# Use timestamp_seconds for better precision handling than from_unixtime
taskMetricsDf = taskMetricsDf.withColumn("ts", timestamp_seconds(col("finishTime") / 1000))

# Add app_id
taskMetricsDf = taskMetricsDf.withColumn("appId", lit(app_id))

#pandas_df = df.toPandas()

try:
    taskMetricsDf.write \
        .format("parquet") \
        .saveAsTable("default.spark_task_metrics")

except Exception:
    taskMetricsDf.write \
        .mode("append") \
        .format("parquet") \
        .saveAsTable("default.spark_task_metrics")


spark.sql("select count(*) from default.spark_task_metrics").show()
