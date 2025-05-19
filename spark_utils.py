import numpy as np
import torch
import transformers
import pandas as pd
import json
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = SparkSession.builder.appName("arxiv_graph") \
    .config("spark.memory.fraction", 0.8) \
    .config("spark.executor.memory", "25g") \
    .config("spark.driver.memory", "25g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "25g") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# JSON lines file
file_path = "/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json"

# Load the JSON lines file
df = spark.read.json(file_path)

# Show the first 5 rows pretty like pandas
df.limit(5).toPandas().head()

# Count max number of authors in each record (comma separated names)
df.withColumn("num_authors", F.size(F.split(F.col("authors"), ", "))).select(F.max("num_authors")).show()

# get set of all unique number of authors in each row
all_authors = df.withColumn("num_authors", F.size(F.split(F.col("authors"), ", "))).select("num_authors").distinct().collect()
all_authors = [x["num_authors"] for x in all_authors]


# Get a file with max authors
df.withColumn("num_authors", F.size(F.split(F.col("authors"), ", "))) \
    .filter(F.col("num_authors") > 2000) \
    .select("authors") \
    .show()

# Use modernbert to save embeddings of every abstract in a vector column
from sentence_transformers import SentenceTransformer
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

model = SentenceTransformer("nomic-ai/modernbert-embed-base")
torch.inference_mode()
model.eval()

# Define a UDF to convert text to vector
def text_to_vector(text):
    # input_ids = model.encode("search_document: " + text, return_tensors='pt')
    with torch.no_grad():
        last_hidden_states = model.encode(["search_document: " + text])
        print(last_hidden_states.shape, flush=True)
    return list(last_hidden_states[0])

# Register the UDF
text_to_vector_udf = udf(text_to_vector, ArrayType(FloatType()))

# Apply the UDF to the abstract column
df = df.withColumn("abstract_vector", text_to_vector_udf(df["abstract"]))
df.show()

# Save the dataframe to parquet
df.write.parquet("/kaggle/working/arxiv_metadata.parquet")

# Read the parquet file
df = spark.read.parquet("/kaggle/working/arxiv_metadata.parquet")
df.show()

# Stop the spark session
spark.stop()
