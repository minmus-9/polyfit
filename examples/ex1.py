#!/usr/bin/env python3

"example usage of the polyfit api"

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

sys.path.insert(0, "..")

from polyfit import PolyfitPlan     ## pylint: disable=wrong-import-position
#from cpolyfit import PolyfitPlan    ## pylint: disable=wrong-import-position

def demo():
    "demo of the api"
    ## pylint: disable=unnecessary-comprehension

    ## poly coefficients to fit, highest degree first
    cv = [2, 1, -1, math.pi]

    ## evaluate the polynomia above using horner's method
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
    plan = PolyfitPlan(len(cv) - 1, xv, wv)
    fit  = plan.fit(yv)

    ## print the fit stats
    deg = fit.maxdeg()
    print("maxdeg", deg)
    print("points", fit.npoints())

    ## print per-degree rms errors
    print("erms  ", [fit.rms_errors()[d] for d in range(deg + 1)])

    ## print a few values
    for i in range(4):
        print("value  %.1f %s" % (xv[i], fit(i, nder=-1)))

    ## coefs about x0=0.
    print("coefs0", fit.coefs(0., deg))
    ## value and all derivs at 0.
    print("value0", fit(xv[0], deg, deg))
    ## coefs halfway through
    print("coefs ", fit.coefs(xv[N >> 1], deg))

if __name__ == "__main__":
    demo()

## EOF
