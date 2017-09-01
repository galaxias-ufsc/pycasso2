# Copyright (c) 2011, Chandler Armstrong (omni dot armstrong at gmail dot com)
# see LICENSE.txt for details
# <http://tixxit.net/2010/03/graham-scan/> provides the basis for the code in
# this file.  the code at that address contains no copyright notice.




"""find the convex hull of a set of points using graham scan"""




from numpy import array, lexsort
from functools import reduce

def _ccw(i, j, k):
    global P
    _P = P
    p_x, p_y = _P[i]
    q_x, q_y = _P[j]
    r_x, r_y = _P[k]
    det = (q_x - p_x) * (r_y - p_y) - (r_x - p_x) * (q_y - p_y)
    return det > 0


def _keep_left(hull, r):
    while len(hull) > 1 and not _ccw(hull[-2], hull[-1], r): hull.pop()
    if not len(hull) or not (hull[-1] == r).all(): hull.append(r)
    return hull


def convexhull(_P):
    """
    Returns an array of the points in convex hull of P in CCW order.

    arguments: P -- a Polygon object or an numpy.array object of points
    """
    global P
    P = _P
    I = lexsort((_P[:,1],_P[:,0]))
    l = reduce(_keep_left, I, [])
    u = reduce(_keep_left, reversed(I), [])
    l.extend(u[i] for i in range(1, len(u) - 1))
    return array(l)
