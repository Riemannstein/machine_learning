import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

print(df.head(5))

y = df.iloc[0:100,4].values

y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, 0:2].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
