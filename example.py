from dask.distributed import Client, LocalCluster
from dask_ml.datasets import make_regression
from dask_ml.linear_model import RidgeRegression


cluster = LocalCluster()
client = Client(cluster)
client

X, y = make_regression(n_samples=10_000_000, chunks=(1000, 1000))

ridge = RidgeRegression(solver="svd")
ridge.fit(X, y)
ridge.predict(X)
ridge.predict(X)
print(ridge.score(X, y))
