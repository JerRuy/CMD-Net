


import numpy as np




def cosine(x,y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    distance = 1 - np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

    return distance



if __name__ == '__main__':

    x=np.random.random(10)
    y=np.random.random(10)
    
    dis = cosine(x,y)
    print(dis)