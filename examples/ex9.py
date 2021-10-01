#!/usr/bin/env python3

"same as ex1, but evaluate the poly in quad precision"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

sys.path.insert(0, "..")

from polyfit import (   ## pylint: disable=wrong-import-position
    PolyfitPlan,
    to_quad, zero, mul, add, to_float
)

def flist(l):
    "format a list to 15 decimal places"
    if not isinstance(l, (list, tuple)):
        l = [l]
    return " ".join("%.15e" % x for x in l)

def demo():
    "demo of the api"
    ## pylint: disable=unnecessary-comprehension,too-many-locals

    ## poly coefficients to fit, highest degree first
    cv   = [2, 1, -1, math.pi]
    cvee = [to_quad(c) for c in cv]

    ## evaluate the polynomial above using horner's method
    def pv(x):
        "evaluate using cv"
        r = zero()
        for c in cvee:
            r = add(mul(r, x), c)
        return r

    ## define the x and y values for the fit
    N  = 10000
    xv = [to_quad(x) for x in range(N)]
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

    ## print per-degree rms errors
    print("erms  ", flist(fit.rms_errors()))

    ## print a few values
    for i in range(4):
        print(
            "value  %.1f %s" % \
            (to_float(xv[i]), flist(ev(to_float(xv[i]), nder=-1)))
        )

    ## print value and all derivatives for all degrees
    for i in range(D + 1):
        print(
            "deg    %d %s" % \
            (i, flist(ev(to_float(to_float(xv[0])), deg=i, nder=-1)))
        )

    ## print coefficients for all degrees about (x - xv[0])
    for i in range(D + 1):
        print("coefs  %d %s" % (i, flist(ev.coefs(to_float(xv[0]), i))))

    ## coefs halfway through
    print()
    print("value ", flist(ev(xv[N>>1])))
    print("exp   ", flist(yv[N>>1]))
    print("coefs ", flist(ev.coefs(to_float(xv[N >> 1]), deg)))

if __name__ == "__main__":
    demo()

## EOF
