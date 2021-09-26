#!/usr/bin/env python3

"cpolyfit demo"

## pylint: disable=invalid-name,bad-whitespace

import array
import sys

sys.path.insert(0, "..")

from cpolyfit import PolyfitPlan

## {{{ demo code
def demo():
    "demo code"
    ## x^2 + 4x + 2
    ## pylint: disable=import-outside-toplevel
    import math
    xv = array.array('d', range(10000))
    def f(x, c=(2, 1, -1, math.pi)):
        "eval test poly"
        r=0.
        for co in c:
            r *= x
            r += co
        return r
    yv = array.array('d', [f(x) for x in xv])
    #wv = array.array('d', [y**-2. for y in yv])
    wv = array.array('d', [1.] * len(xv))

    plan = PolyfitPlan(3, xv, wv)
    fit  = plan.fit(yv)
    ev   = fit.evaluator()

    print("maxdeg  ", plan.maxdeg())
    print("npoints ", plan.npoints())
    print("derivs  ", ev(0., nder=-1))
    print("coefs   ", ev.coefs(0.))
    for x, y in zip(xv[:5], yv):
        p = ev(x, nder=0)
        print(x, y, p)
    for i in range(1 + plan.maxdeg()):
        print("erms    ", i, fit.rms_errors()[i])
    maxrel = -1
    for x, exp in zip(xv, yv):
        obs = ev(x, nder=0)
        rel = abs(obs / exp - 1.)
        if rel > maxrel:
            maxrel = rel
    print("erel    ", maxrel)
    print("free")
    del ev
    del fit
    del plan
## }}}

if __name__ == "__main__":
    demo()
