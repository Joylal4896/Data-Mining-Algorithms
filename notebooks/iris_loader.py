import requests
import numpy as np
import pandas as pd

def load_iris_dataframe():
    IRIS_FILE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    iris_file = requests.get(IRIS_FILE_URL).text
    words = [word.split(",")
               for word in iris_file.split("\n") 
               if len(word) > 0]
    iris_dataframe = pd.DataFrame(words, columns=["A", "B", "C", "D", "category"])
    return iris_dataframe