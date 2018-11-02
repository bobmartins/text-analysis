# Databricks notebook source
import pandas as pd
import numpy as np
import pickle

run_cluster = False
cuda = False
perplexities_w2v = [10.0, 12.5, 20.0]
perplexities_d2v = [2.0, 4.0, 6.0, 8.0, 10.0]
base_path = "/dbfs/mnt/data-shared/derived/recommendations"

# COMMAND ----------

# MAGIC %md # Data Load

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import *

toArray = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

tour_vectors = spark.read.table("recommendation.tour_w2v_clickthrough_vector").withColumn("vector_array", toArray(col("vector")))
tour_info = spark.read.parquet("/mnt/data-shared/derived/recommendations/tour_text").join(spark.read.table("dwh_dim_tour").select("tour_id", "tour_title"), ["tour_id"])
tour_data = tour_info.join(tour_vectors, ["tour_id"])

# COMMAND ----------

# MAGIC %md # Create Doc2Vec Embedding and Clustering

# COMMAND ----------

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tour_data_pd = tour_data.toPandas().drop_duplicates("tour_id")
tour_data_pd['documents'] = tour_data_pd.apply(lambda doc: TaggedDocument(doc.tokens, [doc.tour_id]), axis=1)
tour_data_pd.shape

# COMMAND ----------

max_epochs = 500
vec_size = 300
alpha = 0.05
NUM_WORKERS = 128

if run_cluster:
    model = Doc2Vec(vector_size=vec_size, 
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=10,
                    dm =1,
                    hs=0,
                    negative=5,
                    workers=NUM_WORKERS)
    model.build_vocab(tour_data_pd.documents.values)
    model.train(tour_data_pd.documents.values, total_examples=tour_data_pd.shape[0], epochs=max_epochs)

# COMMAND ----------

if run_cluster:
    model.save(base_path + '/d2v_tourtext_3.pck')
    tour_data_pd['d2v'] = tour_data_pd.tokens.apply(lambda t: model.infer_vector(t))
    tour_data_pd.to_pickle(base_path + '/tour_data_w_d2v.pck')
else:
    model = Doc2Vec.load(base_path + '/d2v_tourtext_3.pck')
    tour_data_pd = pd.read_pickle(base_path + '/tour_data_w_d2v.pck')

# COMMAND ----------

run_cluster=True

# COMMAND ----------

from MulticoreTSNE import MulticoreTSNE as TSNE
tsne_d2v = []
if run_cluster:
  for perp in perplexities_d2v:
      print("running d2v model for perplexity {:.2f}\n".format(perp))
      tsne_model = TSNE(n_components=2, verbose = 2, perplexity=perp, learning_rate=300.0, init='random',
                        early_exaggeration=10.0, n_iter=4000, metric='cosine', n_jobs=16)
      tsne_d2v.append((perp, tsne_model.fit_transform(np.array(tour_data_pd.d2v.tolist()))))
  with open(base_path + '/tsne_d2v_2_to_10.pcl', "wb") as dest:
      pickle.dump(tsne_d2v, dest)
else:
  with open(base_path + '/tsne_d2v_2_to_10.pcl', "rb") as source:
    tsne_d2v = pickle.load(source)

# COMMAND ----------

# MAGIC %md # Customer Journey Clustering

# COMMAND ----------

tsne_w2v = []
if run_cluster:
  for perp in perplexities_w2v:
      print("running d2v model for perplexity {:.2f}\n".format(perp))
      tsne_model = TSNE(n_components=2, verbose = 2, perplexity=perp, learning_rate=300.0, init='random',
                        early_exaggeration=50.0, n_iter=5000, metric='cosine', n_jobs=16)
      tsne_w2v.append((perp, tsne_model.fit_transform(np.array(tour_data_pd.vector_array.tolist()))))
  with open(base_path + '/tsne_w2v_10_12_20.pcl', "wb") as dest:
      pickle.dump(tsne_w2v, dest)
else:
  with open(base_path + '/tsne_w2v_10_12_20.pcl', "rb") as source:
      tsne_w2v = pickle.load(source)

# COMMAND ----------

tsne_w2v[0][1].shape