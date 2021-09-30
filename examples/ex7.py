#!/usr/bin/env python3

"demo integral of fit poly"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import sys

sys.path.insert(0, "..")
import polyfit  as p ## pylint: disable=wrong-import-position
import polyplus as q ## pylint: disable=wrong-import-position

def demo():
    "demo code"
    cv = [3, 2, 1]
    def pv(x):
        "evaluate the polynomial cv[]"
        r = 0.
        for c in cv:
            r *= x
            r += c
        return r
    D  = len(cv) - 1
    N  = 4
    sc = 1.
    xv = [x * sc for x in range(N)]
    yv = [pv(x) for x in xv]
    wv = [1. for _ in xv]

    plan  = p.PolyfitPlan(D, xv, wv)
    fit   = plan.fit(yv)
    integ = q.PolyplusIntegrator(fit, D)
    print(integ.qcoefs())
    ## should be [(1, 0), (1, 0), (1, 0), (0, 0)]

    print([integ(x) for x in xv])
    ## should be [0, 3, 14, 39]

if __name__ == "__main__":
    demo()

## EOF
