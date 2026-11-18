from pyspark.sql import SparkSession, functions as F


spark = SparkSession.builder.appName("FraudTimeAmountAggregates").getOrCreate()
frame = spark.read.option("header", True).csv("hdfs:///fraud/raw/")
frame = frame.withColumn("hour", (F.col("Time").cast("double") / 3600) % 24)
frame = frame.withColumn("Class", F.col("Class").cast("int"))
frame.groupBy("hour", "Class").agg(F.count("*").alias("transactions"), F.avg("Amount").cast("double").alias("average_amount")).orderBy("hour", "Class").write.mode("overwrite").csv("hdfs:///fraud/processed/hourly", header=True)
