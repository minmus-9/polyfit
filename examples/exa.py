#!/usr/bin/env python3

"example usage of polyfit serialization"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import json
import math
import sys

sys.path.insert(0, "..")

from polyfit import (   ## pylint: disable=wrong-import-position
    PolyfitPlan, PolyfitFit, to_quad, to_float, add, mul, zero
)

def flist(l):
    "format a list to 15 decimal places"
    if not isinstance(l, (list, tuple)):
        l = [l]
    return " ".join("%.15e" % x for x in l)

def demo():
    "demo of the api"
    ## pylint: disable=unnecessary-comprehension

    ## poly coefficients to fit, highest degree first
    cv = [to_quad(c) for c in [2, 1, -1, math.pi]]

    ## evaluate the polynomial above using horner's method
    def pv(x):
        "evaluate using cv"
        r = zero()
        for c in cv:
            r = add(mul(r, x), c)
        return r

    ## define the x and y values for the fit
    N  = 100000
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
    fit1 = plan.fit(yv)
    ev1  = fit1.evaluator()

    ## serialize and deserialize
    data = json.dumps(fit1.to_data())
    print("json   %d bytes" % len(data))
    print()
    fit2 = PolyfitFit.from_data(json.loads(data))
    ev2  = fit2.evaluator()

    idx = N >> 1
    print("value1 %.1f %s" % (to_float(xv[idx]), flist(ev1(xv[idx], nder=0))))
    print("value2 %.1f %s" % (to_float(xv[idx]), flist(ev2(xv[idx], nder=0))))
    print()

    ## compare coefs
    x = "  ".join(flist(q) for q in ev1.coefs(xv[0], D))
    print("coefs  ", x)
    x = "  ".join(flist(q) for q in ev2.coefs(xv[0], D))
    print("coefs  ", x)

if __name__ == "__main__":
    demo()

## EOF
