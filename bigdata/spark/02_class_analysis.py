from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("FraudClassAnalysis").getOrCreate()
frame = spark.read.option("header", True).csv("hdfs:///fraud/raw/")
counts = frame.groupBy("Class").count().collect()
print("class_counts", {str(row["Class"]): row["count"] for row in counts})
