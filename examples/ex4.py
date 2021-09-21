#!/usr/bin/env python3

"cpolyfit demo"

## pylint: disable=invalid-name,bad-whitespace

import array
import sys

sys.path.insert(0, "..")

from cpolyfit import Polyfit

## {{{ demo code
def demo():
    "demo code"
    ## x^2 + 4x + 2
    ## pylint: disable=import-outside-toplevel
    import math
    xv = array.array('d', range(100000))
    def f(x, c=(2, 1, -1, math.pi)):
        "eval test poly"
        r=0.
        for co in c:
            r *= x
            r += co
        return r
    yv = array.array('d', [f(x) for x in xv])
    wv = array.array('d', [y**-2. for y in yv])

    pf = Polyfit(3, xv, yv, wv)

    print("maxdeg  ", pf.maxdeg())
    print("npoints ", pf.npoints())
    print("fit time", pf.runtime())
    print("derivs  ", pf(0., nderiv=-1))
    print("coefs   ", pf.coefs())
    for x, y in zip(xv[:5], yv):
        p = pf(x, nderiv=0)
        print(x, y, p)
    for i in range(1 + pf.maxdeg()):
        print("erms    ", i, pf.rms_err(i))
    maxrel = -1
    for x, exp in zip(xv, yv):
        obs = pf(x, nderiv=0)
        rel = abs(obs / exp - 1.)
        if rel > maxrel:
            maxrel = rel
    print("erel    ", maxrel)
    print("free")
    pf.close()
    print("done")
## }}}

if __name__ == "__main__":
    demo()
