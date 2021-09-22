#!/usr/bin/env python3

"example usage of the polyfit api"

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

sys.path.insert(0, "..")

from polyfit import Polyfit ## pylint: disable=wrong-import-position
#from cpolyfit import Polyfit ## pylint: disable=wrong-import-position

def demo():
    "demo of the api"
    ## pylint: disable=unnecessary-comprehension

    ## poly coefficients to fit
    cv = [2, 1, -1, math.pi]

    ## evaluate using horner's method
    def pv(x):
        "evaluate using cv"
        r = 0.
        for c in cv:
            r *= x
            r += c
        return r

    ## define the x and y values for the fit
    xv = [x for x in range(10000)]
    yv = [pv(x) for x in xv]

    ## weights:
    ##     uniform, minimize max residual
    #wv = [1. for _ in xv]

    ##     relative, minimize relative residual
    ##     note that y is nonzero for this example
    wv = [y ** -2. for y in yv]

    ## perform the fit
    fit = Polyfit(len(cv) - 1, xv, yv, wv)

    ## print fit stats
    deg = fit.maxdeg()
    print("maxdeg", deg)
    print("points", fit.npoints())
    print("time  ", fit.runtime())

    ## print per-degree rms errors
    print("erms  ", [fit.rms_err(d) for d in range(deg + 1)])
    ## print relative resid error across all x
    print("relerr", fit.rel_err())

    ## print some values
    for i in range(5):
        print("value  %.1f %s" % (xv[i], fit(i, nderiv=-1)))

    ## coefs about x0=0.
    print("coefs0", fit.coefs(deg, x0=0.))
    ## value and all derivs at 0.
    print("value0", fit(xv[0], deg, deg))

if __name__ == "__main__":
    demo()

## EOF
