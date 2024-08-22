import numpy as np

def cosine(x,y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    distance = 1 - np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return distance
