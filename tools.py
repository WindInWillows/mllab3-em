import numpy as np


# p1 and p2 are []
def get_dis(p1, p2):
    sum = 0.
    for i in xrange(len(p1)):
        sum += (p1[i]-p2[i])**2
    return sum

def equals(p1, p2):
    for i in range(len(p1)):
        if p1[i] - p2[i] < 1e-10:
            return False
    return True


