#!/usr/bin/env python3

"""
polyfit vs numpy relative error in coefs for
uniform and relative weights
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

from np import npfit

sys.path.insert(0, "..")

from polyfit import (   ## pylint: disable=wrong-import-position,wrong-import-order
    PolyfitPlan, add, mul, to_quad, to_float, zero
)

def flist(l):
    "format a list to 15 decimal places"
    if not isinstance(l, list):
        l = [l]
    return " ".join("%.15e" % x for x in l)

def doit(xv, yv, wv, coefs):
    "fit using polyfit and numpy"
    ## perform the fit
    D    = len(coefs) - 1
    plan = PolyfitPlan(D, xv, wv)
    fit  = plan.fit(yv)
    ev   = fit.evaluator()

    ## print the fit stats
    deg = plan.maxdeg()
    print("maxdeg", deg)
    print("points", plan.npoints())

    ## print rel errs in coefs
    obs = ev.coefs(xv[0], -1)
    print("polyfit:")
    print("  exp   ", flist(coefs))
    print("  coefs ", flist(obs))
    rel = [abs(o/e - 1.) if e else 0 for o, e in zip(obs, coefs)]
    print("  relerr", flist(rel))

    ## numpy
    obs = npfit(xv, yv, wv, D)
    rel = [abs(o/e - 1.) if e else 0 for o, e in zip(obs, coefs)]
    print("numpy:")
    print("  exp   ", flist(coefs))
    print("  coefs ", flist(obs))
    print("  relerr", flist(rel))

def demo():
    "compute rel err in poly coefs"
    ## pylint: disable=unnecessary-comprehension

    ## poly coefficients to fit, highest degree first
    cv = [to_quad(c) for c in \
            [0, math.sqrt(2), math.exp(1), math.pi, 1]
         ]
    ceevee = [to_float(c) for c in cv]

    ## evaluate in quad-precision
    def pv(x):
        "evaluate using cv in extended precision"
        x = to_quad(x)
        r = zero()
        for c in cv:
            r = add(mul(r, x), c)
        return to_float(r)

    ## define the x and y values for the fit
    N  = 10000
    xv = [x for x in range(N)]
    yv = [pv(x) for x in xv]

    ## weights:
    ##     uniform to minimize the max residual
    wv = [1. for _ in xv]

    print("uniform weights")
    doit(xv, yv, wv, ceevee)
    print()

    ##     relative to minimize the relative residual
    ##     note that y is nonzero for this example
    wv = [y ** -2. for y in yv]

    print("relative weights")
    doit(xv, yv, wv, ceevee)

if __name__ == "__main__":
    demo()

## EOF
