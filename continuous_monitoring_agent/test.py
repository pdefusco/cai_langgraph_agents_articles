AGENT_NAME = "spark_perf_agent"
AGENT_VERSION = "v1"
POLL_INTERVAL_SECONDS = 30

SPARK_METRICS_TABLE = "default.spark_task_metrics"
CHECKPOINT_TABLE = "agent_checkpoints"


spark.sql('drop table agent_checkpoints')

def create_checkpoint(spark):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CHECKPOINT_TABLE} (
            agent_name STRING,
            agent_version STRING,
            last_job_launch_time TIMESTAMP,
            last_spark_application_id STRING,
            updated_at TIMESTAMP
        )
        PARTITIONED BY (agent_name, agent_version)
        STORED AS PARQUET
    """)

def load_checkpoint(spark):
    df = spark.sql(f"""
        SELECT last_job_launch_time, last_spark_application_id
        FROM {CHECKPOINT_TABLE}
        WHERE agent_name = '{AGENT_NAME}'
          AND agent_version = '{AGENT_VERSION}'
        ORDER BY updated_at DESC
        LIMIT 1
    """)

    if df.count() == 0:
        return None, None

    r = df.collect()[0]
    return r.last_job_launch_time, r.last_spark_application_id


def save_checkpoint(spark, launch_time, app_id):
    spark.sql(f"""
        INSERT INTO {CHECKPOINT_TABLE}
        PARTITION (agent_name='{AGENT_NAME}', agent_version='{AGENT_VERSION}')
        VALUES (
            CAST('{launch_time}' AS TIMESTAMP,
            '{app_id}',
            TIMESTAMP('{datetime.utcnow()}')
        )
    """)

create_checkpoint(spark)
last_launch_time, last_app_id = load_checkpoint(spark)


where_clause = ""
if last_launch_time:
    where_clause = f"""
    WHERE (
        TIMESTAMP(ts) > TIMESTAMP('{last_launch_time}')
        OR (
            TIMESTAMP(ts) = TIMESTAMP('{last_launch_time}')
            AND appId > '{last_app_id}'
        )
    )
    """

df = spark.sql(f"""
    SELECT *
    FROM {SPARK_METRICS_TABLE}
    {where_clause}
    ORDER BY ts, appId
""")

if df.count() > 0:
  rows = [r.asDict() for r in df.collect()]
  #graph.invoke({"metrics": rows, "agent_version": AGENT_VERSION})

  last = rows[-1]
  print(last)

save_checkpoint(spark, last["ts"], last["appId"])

df.show()
