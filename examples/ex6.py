#!/usr/bin/env python3

"""
polyfit vs numpy relative error in coefs for
uniform and relative weights
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math

import testlib as tl

# chofit = tl.n.chofit
chofit = tl.c.chofit


def doit(xv, yv, wv, coefs):
    "fit using polyfit and numpy"
    ## perform the fit
    D = len(coefs) - 1
    plan = tl.p.PolyfitPlan(D, xv, wv)
    fit = plan.fit(yv)
    ev = fit.evaluator()

    ## print the fit stats
    deg = plan.maxdeg()
    print("maxdeg", deg)
    print("points", plan.npoints())

    ## print rel errs in coefs
    obs = tl.dvec(ev.coefs(xv[0], -1))
    cof = tl.dvec(coefs)
    print("polyfit:")
    print("  exp   ", tl.format_list(cof))
    print("  coefs ", tl.format_list(obs))
    rel = [abs(o / e - 1.0) if e else 0 for o, e in zip(obs, cof)]
    print("  relerr", tl.format_list(rel))

    ## numpy
    obs = tl.dvec(chofit(xv, yv, wv, D))
    rel = [abs(o / e - 1.0) if e else 0 for o, e in zip(obs, cof)]
    print("numpy:")
    print("  exp   ", tl.format_list(cof))
    print("  coefs ", tl.format_list(obs))
    print("  relerr", tl.format_list(rel))


def demo():
    "compute rel err in poly coefs"

    ## poly coefficients to fit, highest degree first
    cv = [0, math.sqrt(2), math.exp(1), math.pi, 1]
    cvee = tl.ddpvec(cv)

    ## define the x and y values for the fit
    N = 10000
    xv = tl.ddpvec(range(N))
    yv = [tl.ddpeval(x, cvee) for x in xv]

    ## weights:
    ##     uniform to minimize the max residual
    wv = [tl.p.one() for _ in xv]

    print("uniform weights")
    doit(xv, yv, wv, cvee)
    print()

    ##     relative to minimize the relative residual
    ##     note that y is nonzero for this example
    wv = tl.ddpvec(tl.p.to_float(y) ** -2.0 for y in yv)

    print("relative weights")
    doit(xv, yv, wv, cvee)


if __name__ == "__main__":
    demo()

## EOF
