from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, StructField, StructType


spark = SparkSession.builder.appName("FraudLoadAndValidate").getOrCreate()
fields = [StructField("Time", DoubleType(), True),
          *[StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)],
          StructField("Amount", DoubleType(), True),
          StructField("Class", DoubleType(), True)]
frame = spark.read.schema(StructType(fields)).option("header", True).csv("hdfs:///fraud/raw/")
print("row_count", frame.count())
print("null_counts", frame.select([__import__("pyspark").sql.functions.sum(col(c).isNull().cast("int")).alias(c) for c in frame.columns]).collect()[0].asDict())
