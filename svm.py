import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import sigmoid_kernel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sb
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import pylab
import sklearn.mixture as mixture
#import pyclustertend
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm

df=pd.read_csv("data2.csv")
feature_cols=['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo',  'valence',]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_df =scaler.fit_transform(df[feature_cols])
indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()

cosine = cosine_similarity(normalized_df)
sig_kernel = sigmoid_kernel(normalized_df)

def svm(name, model_type):
    df=pd.read_csv("data2.csv")
    feature_cols=['acousticness', 'danceability', 'duration_ms', 'energy',
                'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                'speechiness', 'tempo',  'valence',]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized_df =scaler.fit_transform(df[feature_cols])
    indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()

    cosine = cosine_similarity(normalized_df)
    sig_kernel = sigmoid_kernel(normalized_df)
    def generate_recommendation(song_title ):

    
        index=indices[song_title]

        score=list(enumerate(model_type[indices[index]]))
    
        similarity_score = sorted(score,key = lambda x:x[1],reverse = True)
    
        similarity_score = similarity_score[1:18]
        
        top_songs_index = [i[0] for i in similarity_score]
        
        top_songs=df.iloc[top_songs_index]
        top_songs= top_songs.assign(Similarity=similarity_score)
        
        
        return top_songs
    a=generate_recommendation(name)
    df2=df.loc[df["song_title"]==name]
    a2= pd.concat([a,df2])
    return a

