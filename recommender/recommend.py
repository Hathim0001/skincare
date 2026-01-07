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

        for i in range(entries):
            for j in range(5, len(self.features)):
                feature = self.features[j]
                if feature in self.df.iloc[i]['concern']:
                    one_hot_encodings[i][j] = 1
        return np.array(one_hot_encodings)

    def wrap(self, info_arr):
        result = {}
        result['brand'] = info_arr[0]
        result['name'] = info_arr[1]
        result['price'] = info_arr[2]
        result['url'] = info_arr[3]
        result['skin type'] = info_arr[4]
        result['concern'] = str(info_arr[5]).split(',')
        result['image_url'] = info_arr[6]
        return result

    def recs_cs(self, vector=None, name=None, label=None, count=4):
        products = []
        if name:
            idx = self.name2index(name)
            fv = self.X[idx]
        elif vector:
            fv = vector

        cs_values = cosine_similarity(np.array([fv, ]), self.X)
        self.df['cs'] = cs_values[0]

        if label:
            dff = self.df[self.df['label'] == label]
        else:
            dff = self.df

        if name:
            dff = dff[dff['name'] != name]

        recommendations = dff.sort_values('cs', ascending=False).head(count)
        data = recommendations[['brand', 'name', 'price', 'url', 'skin type', 'concern', 'image url']].to_dict('split')['data']
        for element in data:
            products.append(self.wrap(element))
        return products
