import numpy as np
import vector

def init(x1,y1,x2,y2):
    return np.array([[x1,y1],[x2,y2]])

def intersect(l1, l2):
    p0,p1 = l1
    p2,p3 = l2
    s1 = p1 - p0
    s2 = p3 - p2
    s = (-s1[1] * (p0[0] - p2[0]) + s1[0] * (p0[1] - p2[1])) / (-s2[0] * s1[1] + s1[0] * s2[1])
    t = ( s2[0] * (p0[1] - p2[1]) - s2[1] * (p0[0] - p2[0])) / (-s2[0] * s1[1] + s1[0] * s2[1])
    i = p0 + (t * s1)
    return i, s, t

def vector(l):
    return l[1] - l[0]

