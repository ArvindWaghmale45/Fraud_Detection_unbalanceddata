$ErrorActionPreference = "Stop"
hdfs dfs -mkdir -p /fraud/raw
hdfs dfs -put -f C:\FraudDetection\ml-service\data\creditcard_transactions.csv /fraud/raw/
hdfs dfs -ls /fraud/raw
