#!/usr/bin/env python3

"example usage of the polyfit api"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace,consider-using-f-string

import math

import testlib as tl  ## pylint: disable=wrong-import-position


def demo():
    "demo of the api"

    ## poly coefficients to fit, highest degree first
    cv = [2, 1, -1, math.pi]

    ## define the x and y values for the fit
    N = 10000
    xv = list(range(N))
    yv = [tl.deval(x, cv) for x in xv]

    ## weights:
    ##     uniform to minimize the max residual
    wv = [1.0] * len(xv)

    ##     relative to minimize the relative residual
    ##     note that y is nonzero for this example
    # wv = [y ** -2. for y in yv]

    ## perform the fit
    D = len(cv) - 1
    plan = tl.p.PolyfitPlan(D, xv, wv)
    fit = plan.fit(yv)
    ev = fit.evaluator()

    ## print the fit stats
    deg = plan.maxdeg()
    print("maxdeg", deg)
    print("points", plan.npoints())

    ## print per-degree rms errors
    print("erms  ", tl.format_list(fit.rms_errors()))

    ## print a few values
    for i in range(4):
        print("value  %.1f %s" % (xv[i], tl.format_list(ev(xv[i], nder=-1))))

    ## print value and all derivatives for all degrees
    for i in range(D + 1):
        print("deg    %d %s" % (i, tl.format_list(ev(xv[0], deg=i, nder=-1))))

    ## print coefficients for all degrees about (x - xv[0])
    for i in range(D + 1):
        print("coefs  %d %s" % (i, tl.format_list(ev.coefs(xv[0], i))))

    ## coefs halfway through
    print("mcoefs", tl.format_list(ev.coefs(xv[N >> 1], deg)))


if __name__ == "__main__":
    demo()

## EOF
