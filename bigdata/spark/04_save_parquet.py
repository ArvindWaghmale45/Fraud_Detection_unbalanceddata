from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("FraudSaveParquet").getOrCreate()
frame = spark.read.option("header", True).inferSchema(True).csv("hdfs:///fraud/raw/")
frame.write.mode("overwrite").parquet("hdfs:///fraud/processed/clean")
