"test support functions"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import sys

import cholesky as c    ## pylint: disable=wrong-import-position
import np as n          ## pylint: disable=wrong-import-position

sys.path.insert(0, "..")
import polyfit as p     ## pylint: disable=wrong-import-position
import polyplus as P    ## pylint: disable=wrong-import-position
import cpolyfit as C    ## pylint: disable=wrong-import-position

def format_list(l):
    "format a list to 15 decimal places"
    if not isinstance(l, (list, tuple)):
        l = [l]
    return " ".join("%23.15e" % x for x in l)

def deval(x, cofs):
    "evaluate the model poly in double prec"
    r = 0.
    for c in cofs:
        r *= x
        r += c
    return r

def qeval(x, cofs):
    "evaluate the model poly in quad prec"
    x = p.to_quad(x)
    r = p.zero()
    for c in cofs:
        r = p.add(p.mul(r, x), c)
    return r

def qevald(x, cofs):
    "evaluate the model poly in quad prec, return float"
    return p.to_float(qeval(x, cofs))

## EOF
