#!/usr/bin/env python3

"same as ex1, but evaluate the poly in quad precision"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math

import testlib as tl

def demo():
    "demo of the api"
    ## pylint: disable=too-many-locals

    ## poly coefficients to fit, highest degree first
    cv   = [2, 1, -1, math.pi]
    cvee = tl.qvec(cv)

    ## define the x and y values for the fit
    N  = 10000
    xv = tl.qvec(range(N))
    yv = [tl.qeval(x, cvee) for x in xv]

    ## weights:
    ##     uniform to minimize the max residual
    wv = [tl.p.one() for _ in xv]

    ## perform the fit
    D    = len(cv) - 1
    plan = tl.p.PolyfitPlan(D, xv, wv)
    fit  = plan.fit(yv)
    ev   = fit.evaluator()

    ## print the fit stats
    deg = plan.maxdeg()
    print("maxdeg", deg)
    print("points", plan.npoints())

    ## print per-degree rms errors
    print("erms  ", tl.format_list(fit.rms_errors()))

    ## print a few values
    for i in range(4):
        print(
            "value  %.1f %s" % \
            (
                tl.p.to_float(xv[i]),
                tl.format_list(ev(tl.p.to_float(xv[i]), nder=-1))
            )
        )

    ## print value and all derivatives for all degrees
    for i in range(D + 1):
        print(
            "deg    %d %s" % \
            (i, tl.format_list(ev(tl.p.to_float(xv[0]), deg=i, nder=-1)))
        )

    ## print coefficients for all degrees about (x - xv[0])
    for i in range(D + 1):
        print(
            "coefs  %d %s" % \
            (i, tl.format_list(ev.coefs(tl.p.to_float(xv[0]), i)))
        )

    ## coefs halfway through
    print()
    print("value ", tl.format_list(ev(xv[N>>1])))
    print("exp   ", tl.format_list(yv[N>>1]))
    print("coefs ", tl.format_list(ev.coefs(tl.p.to_float(xv[N >> 1]), deg)))

if __name__ == "__main__":
    demo()

## EOF
