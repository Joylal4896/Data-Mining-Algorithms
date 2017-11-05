import numpy as np
import pandas as pd

def scale_iris_dataframe(iris_dataframe):
    z_scaler = lambda column: (column - column.mean()) / column.std()
    scaled_iris_dataframe = iris_dataframe[["A", "B", "C", "D"]].apply(pd.to_numeric).apply(z_scaler, axis=0)
    scaled_iris_dataframe["category"] = iris_dataframe["category"]
    return scaled_iris_dataframe