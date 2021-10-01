#!/usr/bin/env python3

"""
relative error in coefs, polyfit vs numpy
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

sys.path.insert(0, "..")

from np      import npfit       ## pylint: disable=wrong-import-position
from polyfit import PolyfitPlan ## pylint: disable=wrong-import-position

def flist(l):
    "format a list to 15 decimal places"
    if not isinstance(l, list):
        l = [l]
    return " ".join("%.15e" % x for x in l)

def demo():
    "compute rel err in poly coefs"
    ## pylint: disable=unnecessary-comprehension,too-many-locals

    ## poly coefficients to fit, highest degree first
    cv = [math.sqrt(2), math.exp(1), math.pi, 1]

    ## evaluate the polynomial above using horner's method
    def pv(x):
        "evaluate using cv"
        r = 0.
        for c in cv:
            r *= x
            r += c
        return r

    ## define the x and y values for the fit
    N  = 10000
    xv = [x for x in range(N)]
    yv = [pv(x) for x in xv]

    ## weights:
    ##     uniform to minimize the max residual
    wv = [1. for _ in xv]

    ##     relative to minimize the relative residual
    ##     note that y is nonzero for this example
    #wv = [y ** -2. for y in yv]

    ## perform the fit
    D    = len(cv) - 1
    plan = PolyfitPlan(D, xv, wv)
    fit  = plan.fit(yv)
    ev   = fit.evaluator()

    ## print the fit stats
    deg = plan.maxdeg()
    print("maxdeg", deg)
    print("points", plan.npoints())

    ## print rel errs in coefs
    obs = ev.coefs(xv[0], -1)
    rel = [abs(o/e - 1.) for o, e in zip(obs, cv)]
    print("polyfit:")
    print("  coefs ", flist(obs))
    print("  relerr", flist(rel))

    ## numpy
    obs = npfit(xv, yv, wv, D)
    rel = [abs(o/e - 1.) for o, e in zip(obs, cv)]
    print("numpy:")
    print("  coefs ", flist(obs))
    print("  relerr", flist(rel))

if __name__ == "__main__":
    demo()

## EOF
