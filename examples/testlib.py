"test support functions"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace,unused-import

import sys

import cholesky as c    ## pylint: disable=wrong-import-position
import np as n          ## pylint: disable=wrong-import-position

sys.path.insert(0, "..")
import polyfit as p     ## pylint: disable=wrong-import-position
import polyplus as q    ## pylint: disable=wrong-import-position
import cpolyfit as C    ## pylint: disable=wrong-import-position

def format_list(l):
    "format a list to 15 decimal places"
    if not isinstance(l, (list, tuple)):
        l = [l]
    ret = [ ]
    for item in l:
        if isinstance(item, (list, tuple)):
            s = "(%23.15e %23.15e)" % item
        else:
            s = "%23.15e" % item
        ret.append(s)
    return " ".join(ret)

def dvec(v):
    "convert vector to float"
    return [p.to_float(x) for x in v]

def qvec(v):
    "convert vector to quad"
    return [p.to_quad(x) for x in v]

def deval(x, cofs):
    "evaluate the model poly in double prec"
    r = 0.
    for cof in cofs:
        r *= x
        r += cof
    return r

def qeval(x, cofs):
    "evaluate the model poly in quad prec"
    x = p.to_quad(x)
    r = p.zero()
    for cof in cofs:
        r = p.add(p.mul(r, x), cof)
    return r

def qevald(x, cofs):
    "evaluate the model poly in quad prec, return float"
    return p.to_float(qeval(x, cofs))

def qx_to_the_k(x, k):
    "compute x**k in quad precision"
    x   = p.to_quad(x)
    ret = p.one()
    while k:
        if k & 1:
            ret = p.mul(ret, x)
        k >>= 1
        x   = p.mul(x, x)
    return ret

def fac_(N):
    "factorial using recursion"
    if N < 2:
        return 1
    return N * fac(N - 1)

FAC = { }

def fac(N):
    "factorial"
    if N not in FAC:
        FAC[N] = fac_(N)
    return FAC[N]

def bincof(N, K):
    "biNomial coefficieNt"
    return fac(N) // fac(K) // fac(N - K)

def itoq(N):
    "integer to quad"
    x = float(N)
    return (x, N - x)

## EOF
