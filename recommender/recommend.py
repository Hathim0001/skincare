import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, df_path, features):
        self.df = pd.read_csv(df_path, index_col='id')
        self.features = features
        self.LABELS = self.df['label'].unique().tolist()
        self.X = self.get_one_hot_encoded_vectors()

    def name2index(self, name):
        return self.df[self.df["name"] == name].index.tolist()[0]

    def get_one_hot_encoded_vectors(self):
        entries = len(self.df)
        one_hot_encodings = np.zeros([entries, len(self.features)])
        for i in range(entries):
            for j in range(5):
                target = self.features[j]
                sk_type = self.df.iloc[i]['skin type']
                if sk_type == 'all':
                    one_hot_encodings[i][0:5] = 1
                elif target == sk_type:
                    one_hot_encodings[i][j] = 1
        return np.array(one_hot_encodings)
