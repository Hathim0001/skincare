import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, df_path, features):
        self.df = pd.read_csv(df_path, index_col='id')
        self.features = features
        self.LABELS = self.df['label'].unique().tolist()

    def name2index(self, name):
        return self.df[self.df["name"] == name].index.tolist()[0]
