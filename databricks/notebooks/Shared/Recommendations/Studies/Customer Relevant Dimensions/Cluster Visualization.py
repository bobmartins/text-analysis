# Databricks notebook source
import pandas as pd
import numpy as np
import pickle

base_path = "/dbfs/mnt/data-shared/derived/recommendations"
d2v_model = 'd2v_tourtext_3.pck'
d2v_tsne = 'tsne_d2v_2_to_10.pcl'
w2v_tsne = 'tsne_w2v_10_12_20.pcl'

# COMMAND ----------

# MAGIC %md # Raw Data Load

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import *

toArray = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

tour_vectors = spark.read.table("recommendation.tour_w2v_clickthrough_vector").withColumn("vector_array", toArray(col("vector")))
tour_info = spark.read.parquet("/mnt/data-shared/derived/recommendations/tour_text").join(spark.read.table("dwh_dim_tour").select("tour_id", "tour_title"), ["tour_id"])
tour_data = tour_info.join(tour_vectors, ["tour_id"])

# COMMAND ----------

from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go

def gen_scatter_traces(points, tsne_points, colors=None):
    cmax = 0
    cmin = 0
    if colors is not None:
      for (i, (perp, tsne_results)) in enumerate(tsne_points):
          yield go.Scattergl(
              x = tsne_results[:,0],
              y = tsne_results[:,1],
              mode = 'markers',
              marker = dict(
                  size = 5,
                  cmax = np.max(colors),
                  cmin = np.min(colors),
                  color=colors,
                  colorscale='Rainbow',                
                  line = dict(
                      width = 1,
                      color = 'rgb(0, 0, 0)'
                  )
              ),
              text=points.apply(lambda x: "{}: {}".format(x.tour_id, x.tour_title), axis=1)
          )
    else:
      for (i, (perp, tsne_results)) in enumerate(tsne_points):
          yield go.Scattergl(
              x = tsne_results[:,0],
              y = tsne_results[:,1],
              mode = 'markers',
              marker = dict(
                  size = 5,
                  color = 'rgba(50, 150, 75, .2)',
                  line = dict(
                      width = 1,
                      color = 'rgb(0, 0, 0)'
                  )
              ),
              text=points.apply(lambda x: "{}: {}".format(x.tour_id, x.tour_title), axis=1)
          )

def scatter(points, tsne_points, height=800, width=800, colors=None):
    layout = go.Layout(
        autosize=False,
        width=width,
        height=height,
        title='tSNE Results for perplexity {:.2f}'.format(tsne_points[0])
    )

    fig = go.Figure(data=[x for x in gen_scatter_traces(points, [tsne_points], colors)], layout=layout)
    return(py.plot(fig, output_type='div'))
       
def scatter_comp(points, tsne_points):
  fig = tools.make_subplots(rows=len(tsne_points), cols=1,
                            subplot_titles=tuple(["Perplexity: {:6.2f}".format(perp) for (perp,_) in tsne_points]))
  for i, trace in enumerate(gen_scatter_traces(points, tsne_points)):
        fig.append_trace(trace, i+1, 1)

  fig['layout'].update(height=1000*len(tsne_points), width=1200, title='tSNE Results for several perplexities')
  return(py.plot(fig, output_type='div'))



# COMMAND ----------

from sklearn.cluster import KMeans
def cluster_this(data, n_clusters, random_state):
  w2v_clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=16).fit(data)
  return(w2v_clustering.predict(data))

# COMMAND ----------

# MAGIC %md # Load Doc2Vec Embedding

# COMMAND ----------

from gensim.models.doc2vec import Doc2Vec

model  = Doc2Vec.load(base_path + '/' + d2v_model)
tour_data_pd = pd.read_pickle(base_path + '/tour_data_w_d2v.pck')
tour_data_pd.shape

# COMMAND ----------

tsne_d2v = []
with open(base_path + '/' + d2v_tsne, "rb") as source:
    tsne_d2v = pickle.load(source)
[x.shape for (_, x) in tsne_d2v]

# COMMAND ----------

#p = scatter_comp(tour_data_pd, tsne_d2v)
#displayHTML(p)

# COMMAND ----------

tour_data_pd['d2v_tsne'] = [np.array(x) for x in tsne_d2v[0][1].tolist()]
tour_data_pd['d2v_cluster'] = cluster_this(np.array(tour_data_pd.d2v_tsne.tolist()), 100, 20)

# COMMAND ----------

p = scatter(tour_data_pd, (20.0, np.array(tour_data_pd.d2v_tsne.tolist())), 800, 1600, tour_data_pd.d2v_cluster)
displayHTML(p)

# COMMAND ----------

# MAGIC %md # Analysing Click Behaviour Word2Vec results

# COMMAND ----------

tsne_w2v = []
with open(base_path + '/' + w2v_tsne, "rb") as source:
    tsne_w2v = pickle.load(source)
[x.shape for (_, x) in tsne_w2v]

# COMMAND ----------

#p = scatter_comp(tour_data_pd, tsne_w2v)
#displayHTML(p)

# COMMAND ----------

tour_data_pd['w2v_tsne'] = [np.array(x) for x in tsne_w2v[2][1].tolist()]

# COMMAND ----------

tour_data_pd['w2v_cluster'] = cluster_this(np.array(tour_data_pd.w2v_tsne.tolist()), 100, 20)

# COMMAND ----------

p = scatter(tour_data_pd, (20.0, np.array(tour_data_pd.w2v_tsne.tolist())), 800, 1600, tour_data_pd.w2v_cluster)
displayHTML(p)

# COMMAND ----------

bookers = spark.read.parquet(base_path + "/reco_study_bookers").toPandas()
non_bookers = spark.read.parquet(base_path + "/reco_study_non_bookers").toPandas()

# COMMAND ----------

w2v_cluster_map = {k: v for (k, v) in tour_data_pd.set_index("tour_id").w2v_cluster.iteritems()}
d2v_cluster_map = {k: v for (k, v) in tour_data_pd.set_index("tour_id").d2v_cluster.iteritems()}

bookers['visited_w2v_clusters'] = bookers.visited_tour_ids.apply(lambda tour_list: [w2v_cluster_map.get(int(t), None) for t in tour_list if int(t) in w2v_cluster_map])
non_bookers['visited_w2v_clusters'] = non_bookers.visited_tour_ids.apply(lambda tour_list: [w2v_cluster_map.get(int(t), None) for t in tour_list if int(t) in w2v_cluster_map])
bookers['visited_d2v_clusters'] = bookers.visited_tour_ids.apply(lambda tour_list: [d2v_cluster_map.get(int(t), None) for t in tour_list if int(t) in d2v_cluster_map])
non_bookers['visited_d2v_clusters'] = non_bookers.visited_tour_ids.apply(lambda tour_list: [d2v_cluster_map.get(int(t), None) for t in tour_list if int(t) in d2v_cluster_map])
bookers = bookers[bookers.visited_w2v_clusters.apply(lambda tl: len(tl) >= 4) & bookers.visited_d2v_clusters.apply(lambda tl: len(tl) >= 4)]
non_bookers = non_bookers[non_bookers.visited_w2v_clusters.apply(lambda tl: len(tl) >= 4) & non_bookers.visited_d2v_clusters.apply(lambda tl: len(tl) >= 4)]

# COMMAND ----------

from itertools import tee

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def cluster_change(cluster_list):
  return [0.0 if first == second else 1.0 for (first, second) in pairwise(cluster_list)]
  
bookers['w2v_cluster_change'] = bookers.visited_w2v_clusters.apply(cluster_change)
bookers['d2v_cluster_change'] = bookers.visited_d2v_clusters.apply(cluster_change)
non_bookers['w2v_cluster_change'] = non_bookers.visited_w2v_clusters.apply(cluster_change)
non_bookers['d2v_cluster_change'] = non_bookers.visited_d2v_clusters.apply(cluster_change)

# COMMAND ----------

from statsmodels.stats.weightstats import ttest_ind

def converged(cluster_changes):
  half_point = len(cluster_changes) // 2
  return np.mean(cluster_changes[:half_point]) > np.mean(cluster_changes[half_point:])

def bookers_converge_more(bookers, non_bookers, cluster_change_col):
  return np.mean(non_bookers[cluster_change_col].apply(converged)) < np.mean(bookers[cluster_change_col].apply(converged))

(_, d2v_p_value, _) = ttest_ind(bookers.d2v_cluster_change.apply(converged).astype(np.float), non_bookers.d2v_cluster_change.apply(converged).astype(np.float), 'larger')
(_, w2v_p_value, _) = ttest_ind(bookers.w2v_cluster_change.apply(converged).astype(np.float), non_bookers.w2v_cluster_change.apply(converged).astype(np.float), 'larger')

print('Based on {}, the hypothesis "Bookers converge more than non-bookers" is {} (p-value: {:.4f}).\n'.format('d2v_cluster_change', bookers_converge_more(bookers, non_bookers, 'd2v_cluster_change'), d2v_p_value))
print('Based on {}, the hypothesis "Bookers converge more than non-bookers" is {} (p-value: {:.4f}).\n'.format('w2v_cluster_change', bookers_converge_more(bookers, non_bookers, 'w2v_cluster_change'), w2v_p_value))