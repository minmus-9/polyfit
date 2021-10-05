#!/usr/bin/env python3

"demo integral of fit poly"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import testlib as tl

def demo():
    "demo code"
    cv = [3, 2, 1]
    D  = len(cv) - 1
    N  = 4
    sc = 1.
    xv = [x * sc for x in range(N)]
    yv = [tl.deval(x, cv) for x in xv]
    wv = [1. for _ in xv]

    plan  = tl.p.PolyfitPlan(D, xv, wv)
    fit   = plan.fit(yv)
    integ = tl.q.PolyplusIntegrator(fit, D)

    ## quick serialization test
    ser   = integ.to_data()
    assert isinstance(ser, dict)
    integ = tl.q.PolyplusIntegrator.from_data(ser)

    print(integ.qcoefs())
    ## should be [(1, 0), (1, 0), (1, 0), (0, 0)]

    print([integ(x) for x in xv])
    ## should be [0, 3, 14, 39]

if __name__ == "__main__":
    demo()

## EOF
