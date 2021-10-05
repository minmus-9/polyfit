#!/usr/bin/env python3

"cpolyfit demo"

## pylint: disable=invalid-name,bad-whitespace

from __future__ import print_function as _

import array
import sys

sys.path.insert(0, "..")

import testlib as tl    ## pylint: disable=wrong-import-position

def demo():
    "demo code"
    ## pylint: disable=import-outside-toplevel,too-many-locals
    import math
    cv = [2, 1, -1, math.pi]
    xv = array.array('d', range(10000))
    yv = array.array('d', [tl.deval(x, cv) for x in xv])
    #wv = array.array('d', [y**-2. for y in yv])
    wv = array.array('d', [1.] * len(xv))

    plan = tl.C.PolyfitPlan(3, xv, wv)
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
    print("done")

if __name__ == "__main__":
    demo()

## EOF
