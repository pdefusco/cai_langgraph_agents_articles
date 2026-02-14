# Placeholder

### Example Spark Submit

```
spark-submit \
  --executor-memory 4g \
  --executor-cores 2 \
  --num-executors 4 \
  --conf spark.sql.shuffle.partitions=200 \
  sparkApp.py
```
