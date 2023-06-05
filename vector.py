import numpy as np

def init(x, y=None):
    return np.array([x,y])

def random(a=0, b=1):
    return np.random.uniform(a,b,size=(2,))

def add(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1]]

def sub(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1]]

def mult(v, k):
    return [v[0] * k, v[1] * k]

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def norm(v):
    return np.sqrt(v[0]**2+v[1]**2)

def normalise(v):
    n = norm(v)
    if n == 0:
        return v
    return [v[0]/n, v[1]/n]

def limit(v, m):
    if norm(v) > m:
        return mult(normalise(v), m)
    return v

def dist(v1, v2):
    return norm(sub(v1,v2))

def dist_quick(v1, v2):
    v = sub(v1, v2)
    return v[0] ** 2 + v[1] ** 2


