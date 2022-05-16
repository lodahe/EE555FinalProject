"""
Author: Austin Kao
Class definition of a quaternion
Not being used in final version of project
"""
import numpy as np


class Quaternion:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def get_distance(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return -1

    def get_conjugate(self):
        return tuple(self.a, -self.b, -self.c, -self.d)

    def __str__(self):
        return "{a}+{b}i+{c}j+{d}k".format(a=self.a, b=self.b, c=self.c, d=self.d)

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(self.a+other.a, self.b+other.b, self.c+other.c, self.d+other.d)

def quaternion_distance(q1, q2):
    return 0

def quaternion_weighted_average_filter(quaternions):
    return np.sum(quaternions) / np.sum(quaternions)
