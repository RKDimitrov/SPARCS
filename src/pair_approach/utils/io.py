import pandas as pd
import numpy as np

def save_vectors_to_csv(vectors, filename):
    df = pd.DataFrame(vectors, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False)

def load_vectors_from_csv(filename):
    df = pd.read_csv(filename)
    return df[['x', 'y', 'z']].values 