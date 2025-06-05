import os
os.environ["PYSPARK_PYTHON"] = r"C:\Python310\python.exe"
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression

# Pornește sesiunea Spark
spark = SparkSession.builder.appName("Flight Delay Prediction").getOrCreate()

# Încarcă fișierul CSV cu întârzieri
df = spark.read.csv("Airline_Delay_Cause.csv", header=True, inferSchema=True)

# Filtrare doar pe coloanele care ne interesează
df = df.select("carrier", "month", "arr_delay").dropna()

# Indexare carrier (string -> numeric)
carrier_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_indexed")
carrier_indexer_model = carrier_indexer.fit(df)
df = carrier_indexer_model.transform(df)

# Vector assembler: combină coloanele numerice într-un singur vector de input
assembler = VectorAssembler(inputCols=["carrier_indexed", "month"], outputCol="features")
df = assembler.transform(df)

# Construim modelul de regresie
lr = LinearRegression(featuresCol="features", labelCol="arr_delay")
model = lr.fit(df)

# Pregătim date noi pentru predicție
sample_data = spark.createDataFrame([
    (2025, 6, "9E", "ABE"),
    (2025, 6, "AA", "JFK"),
    (2025, 6, "AS", "BOS"),
    (2025, 7, "9E", "ABE"),
    (2025, 7, "AA", "JFK"),
    (2025, 7, "AS", "BOS"),
], ["year", "month", "carrier", "airport"])

# Aplicăm același indexer și assembler pe datele noi
sample_data = carrier_indexer_model.transform(sample_data)
sample_data = assembler.transform(sample_data)

# Predicție
predictions = model.transform(sample_data)
predictions.select("carrier", "month", "prediction").show()

# Oprește sesiunea Spark
spark.stop()
