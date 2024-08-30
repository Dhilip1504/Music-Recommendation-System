import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression

train = pd.read_csv('train.csv')
songs = pd.read_csv('songs.csv')
sfl = pd.read_csv('save_for_later.csv')
song_labels = pd.read_csv('song_labels.csv')
test = pd.read_csv('test.csv')

cust_avg = train.groupby("customer_id")["score"].mean()
song_avg = train.groupby("song_id")["score"].mean()
cust_count = train.groupby("customer_id")["score"].count()
song_count = train.groupby("song_id")["score"].count()

customer_dict = {}
for i, cust in enumerate(train["customer_id"].unique()):
    customer_dict[cust] = i

comments = [0]*10001
for index, row in songs.iterrows():
    s_id = row["song_id"]
    comments[int(s_id)] = float(row["number_of_comments"])

X = []
Y = []
weights = []
X_1 = []
X_2 = []
X_3 = []
for index, row in train.iterrows():
    c_id = row["customer_id"]
    s_id = row["song_id"]
    x1 = cust_avg[c_id]
    x2 = song_avg[int(s_id)]
    x3 = comments[int(s_id)]
    weights.append(1/(cust_count[c_id] + song_count[int(s_id)]))
    X_1.append(x1)
    X_2.append(x2)
    X_3.append(x3)
    X.append([x1, x2, x3])
    Y.append(float(row['score']))

X, Y = np.array(X), np.array(Y)
model = LinearRegression().fit(X, Y, weights)

score = []
for index, row in test.iterrows():
    c_id = row["customer_id"]
    s_id = row["song_id"]
    xi = np.array([cust_avg[c_id], song_avg[int(s_id)],
                   comments[int(s_id)]]).reshape(1, 3)
    s = min(round((model.predict(xi)[0]), 4), 5)
    score.append(s)

df = {"score": score}
predictions = pd.DataFrame(df, columns=['score'])
predictions.index.name = "test_row_id"

predictions.to_csv(
    "ce18b010_ee17b051.csv")
