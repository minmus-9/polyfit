#!/usr/bin/env python3

"""
relative error in coefs, polyfit vs numpy
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

sys.path.insert(0, "..")

from polyfit import (   ## pylint: disable=wrong-import-position
    PolyfitPlan, to_float, to_quad, zero, one, add, mul
)

from cholesky import chofit     ## pylint: disable=wrong-import-position
#from np import chofit           ## pylint: disable=wrong-import-position

def flist(l):
    "format a list to 15 decimal places"
    if not isinstance(l, (list, tuple)):
        l = [l]
    return " ".join("%.15e" % x for x in l)

def demo():
    "compute rel err in poly coefs"
    ## pylint: disable=unnecessary-comprehension,too-many-locals

    ## poly coefficients to fit, highest degree first
    cv   = [2, 1, -1, math.pi]
    cvee = [to_quad(z) for z in cv]
    #cv   = [to_quad(z) for z in [math.sqrt(2), math.exp(1), math.pi, 1]]

    ## evaluate the polynomial above using horner's method
    def pv(x):
        "evaluate using cv"
        x = to_quad(x)
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
    wv = [one() for _ in xv]

    ## perform the fit
    D    = len(cv) - 1
    plan = PolyfitPlan(D, xv, wv)
    fit  = plan.fit(yv)
    ev   = fit.evaluator()

    ## print rel errs in coefs
    obs = [to_float(c) for c in ev.coefs(xv[0], -1)]
    rel = [abs(o/e - 1.) if e else o for o, e in zip(obs, cv)]
    print("polyfit:")
    print("  coefs ", flist(obs))
    print("  relerr", flist(rel))
    print("    P(0)", flist(ev(zero())))
    print(" P(N>>1)", flist(ev(to_quad(N >> 1))))
    print("     exp", flist(pv(N >> 1)))

    ## numpy
    cof = chofit(xv, yv, wv, D)
    def nv(x):
        x = to_quad(x)
        r = zero()
        for c in cof:
            r = add(mul(r, x), c)
        return r

    obs = [to_float(c) for c in cof]
    rel = [abs(o/e - 1.) if e else o for o, e in zip(obs, cv)]
    print("numpy:")
    print("  coefs ", flist(obs))
    print("  relerr", flist(rel))
    print("    P(0)", flist(nv(0)))
    print(" P(N>>1)", flist(nv(N >> 1)))
    print("     exp", flist(pv(N >> 1)))

if __name__ == "__main__":
    demo()

## EOF
