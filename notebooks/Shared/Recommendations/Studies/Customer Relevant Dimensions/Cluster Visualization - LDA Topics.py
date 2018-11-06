# Databricks notebook source
import pandas as pd
import numpy as np
import pickle
from statsmodels.stats.weightstats import ttest_ind

base_path = "/dbfs/mnt/data-shared/derived/recommendations"
lda_tsne = '/andre/tsne_lda.pcl'

from itertools import tee

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def cluster_change(cluster_list):
  return [0.0 if first == second else 1.0 for (first, second) in pairwise(cluster_list)]

def converged(cluster_changes):
  half_point = len(cluster_changes) // 2
  return np.mean(cluster_changes[:half_point]) > np.mean(cluster_changes[half_point:])

def bookers_converge_more(bookers, non_bookers, cluster_change_col):
  return np.mean(non_bookers[cluster_change_col].apply(converged)) < np.mean(bookers[cluster_change_col].apply(converged))


# COMMAND ----------

# MAGIC %sh
# MAGIC ls -la /dbfs/mnt/data-shared/derived/recommendations/andre

# COMMAND ----------

# MAGIC %md # Raw Data Load

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import *

toArray = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

tour_data_pd = pd.read_pickle(base_path + '/andre/joined_vectors_pandas.pcl')
displayHTML(tour_data_pd.to_html(max_rows=20))

# COMMAND ----------

bookers = spark.read.parquet(base_path + "/reco_study_bookers").toPandas()
non_bookers = spark.read.parquet(base_path + "/reco_study_non_bookers").toPandas()

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
    data_clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=16).fit(data)
    return(data_clustering.predict(data))

# COMMAND ----------

# MAGIC %md # Analysing Click Behaviour LDA results

# COMMAND ----------

tsne_lda = []
with open(base_path + lda_tsne, "rb") as source:
    tsne_lda = pickle.load(source)
[x.shape for (_, x) in tsne_lda]

# COMMAND ----------

#p = scatter_comp(tour_data_pd, tsne_w2v)
#displayHTML(p)

# COMMAND ----------

tour_data_pd['lda_tsne'] = [np.array(x) for x in tsne_lda[0][1].tolist()]

# COMMAND ----------

tour_data_pd['lda_cluster'] = cluster_this(np.array(tour_data_pd.lda_tsne.tolist()), 30, 20)

# COMMAND ----------

p = scatter(tour_data_pd, (20.0, np.array(tour_data_pd.lda_tsne.tolist())), 800, 1600, tour_data_pd.lda_cluster)
displayHTML(p)

# COMMAND ----------

# MAGIC %md # LDA Hypothesis Test

# COMMAND ----------

lda_cluster_map = {k: v for (k, v) in tour_data_pd.set_index("tour_id").lda_cluster.iteritems()}

bookers['visited_lda_clusters'] = bookers.visited_tour_ids.apply(lambda tour_list: [lda_cluster_map.get(int(t), None) for t in tour_list if int(t) in lda_cluster_map])
non_bookers['visited_lda_clusters'] = non_bookers.visited_tour_ids.apply(lambda tour_list: [lda_cluster_map.get(int(t), None) for t in tour_list if int(t) in lda_cluster_map])
bookers = bookers[bookers.visited_lda_clusters.apply(lambda tl: len(tl) >= 4)]
non_bookers = non_bookers[non_bookers.visited_lda_clusters.apply(lambda tl: len(tl) >= 4)]
bookers['lda_cluster_change'] = bookers.visited_lda_clusters.apply(cluster_change)
non_bookers['lda_cluster_change'] = non_bookers.visited_lda_clusters.apply(cluster_change)
(_, lda_p_value, _) = ttest_ind(bookers.lda_cluster_change.apply(converged).astype(np.float), non_bookers.lda_cluster_change.apply(converged).astype(np.float), 'larger')

displayHTML('<p>Based on <code>{}</code>, the hypothesis <i>"Bookers converge more than non-bookers"</i> is <b>{}</b> (p-value: {:.4f}).</p>'.format(
            'lda_cluster_change', bookers_converge_more(bookers, non_bookers, 'lda_cluster_change'), lda_p_value))