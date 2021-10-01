"numpy fit using quad-precision setup"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

import numpy as np

sys.path.insert(0, "..")
from polyfit import (   ## pylint: disable=wrong-import-position
    mul, vectorsum, vappend, to_quad, to_float
)

def npfit(xv, yv, wv, D):
    "numpy fit, quad-precision setup"
    ## pylint: disable=too-many-locals
    xv = [to_quad(x) for x in xv]
    yv = [to_quad(y) for y in yv]
    wv = [to_quad(w) for w in wv]
    xa = wv[:]          ## accumulator
    mx = [ ]            ## moments
    b  = [ ]            ## rhs in ac=b
    for i in range((D + 1) * 2):
        if i <= D:
            v = [ ]
            for x, y in zip(xa, yv):
                vappend(v, mul(x, y))
            b.append(vectorsum(v))
        v = [ ]
        for x in xa:
            vappend(v, x)
        mx.append(to_float(vectorsum(v)))
        for j, x in enumerate(xa):
            xa[j] = mul(x, xv[j])
    b  = np.array([to_float(x) for x in b])
    mx = np.array(mx)
    ## build the normal matrix
    a  = [ ]
    for i in range(D + 1):
        a.append(mx[i:i+D+1])
    a = np.array(a)
    ## solve the normal equations
    L    = np.linalg.cholesky(a)
    y    = np.linalg.solve(L, b)
    cofs = np.linalg.solve(np.transpose(L), y)
    cofs = list(cofs)
    cofs.reverse()
    return cofs

def demo():
    "numpy demo"
    ## pylint: disable=unnecessary-comprehension
    cv = [2, 1, -1, math.pi]
    def pv(x):
        "evaluate the model poly"
        r = 0.
        for c in cv:
            r *= x
            r += c
        return r
    N  = 100000
    D  = len(cv) - 1
    xv = [x for x in range(N)]
    yv = [pv(x) for x in xv]
    wv = [1. for _ in xv]
    print(npfit(xv, yv, wv, D))

if __name__ == "__main__":
    demo()

## EOF
