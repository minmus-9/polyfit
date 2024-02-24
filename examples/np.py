"numpy fit using DDP setup"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import sys

import numpy as np
import scipy.linalg as la

sys.path.insert(0, "..")
from polyfit import (   ## pylint: disable=wrong-import-position
    mul, vectorsum, vappend, to_ddp, to_float
)

def chofit(xv, yv, wv, D):
    "cholesky numpy fit, DDP setup"
    ## pylint: disable=too-many-locals
    xv = [to_ddp(x) for x in xv]
    yv = [to_ddp(y) for y in yv]
    wv = [to_ddp(w) for w in wv]
    xa = wv[:]          ## accumulator
    mx = [ ]            ## DDP moments
    r  = [ ]            ## DDP rhs in Ac=r
    for i in range((D + 1) * 2):
        if i <= D:
            ## compute rhs
            v = [ ]
            for x, y in zip(xa, yv):
                vappend(v, mul(x, y))
            r.append(vectorsum(v))
        ## compute moments up to 2D+1
        v = [ ]
        for x in xa:
            vappend(v, x)
        mx.append(vectorsum(v))
        for j, x in enumerate(xa):
            xa[j] = mul(x, xv[j])
    ## build the normal matrix from qprec moments
    A = [ ]
    for i in range(D + 1):
        A.append([to_float(m) for m in mx[i:i+D+1]])
    A = np.array(A)
    ## build the numpy rhs from the qprec one
    b = np.array([to_float(x) for x in r])
    ## solve the normal equations
    info = la.cho_factor(A)
    cofs = la.cho_solve(info, b)
    ## get 'em into std order for horner's method
    cofs = [to_ddp(c) for c in cofs]
    cofs.reverse()
    return cofs

## EOF
