#!/usr/bin/env python3

"""
pushing the envelope for polyfit and numpy
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

sys.path.insert(0, "..")

import polyfit as p             ## pylint: disable=wrong-import-position

from cholesky import chofit ## pylint: disable=wrong-import-position
#from np import chofit       ## pylint: disable=wrong-import-position

def flist(l):
    "format a list to 16 sigfigs"
    if not isinstance(l, (list, tuple)):
        l = [l]
    return " ".join("%23.15e" % x for x in l)

def demo():
    # pylint: disable=too-many-statements,using-constant-test
    """
    this demo shows the limits of polyfit and numpy
    for various numbers of data points
    """
    ## pylint: disable=unnecessary-comprehension,too-many-locals

    ## "limit" means the estimation of pi
    ## is worse than 3.14159

    if 0:
        N = 10
        D = 10      ## polyfit and cho handle every degree
        D = 7       ## numpy limit
    elif 0:
        N = 100
        D = 16      ## polyfit limit
        D = 9       ## cho limit
        D = 4       ## numpy limit
    elif 0:
        N = 1000
        D = 10      ## polyfit limit
        D = 7       ## cho limit
        D = 3       ## numpy limit
    elif 1:
        N = 10000
        D = 7       ## polyfit limit
        D = 5       ## cho limit
        D = 2       ## numpy limit
    elif 0:
        N = 100000
        D = 6       ## polyfit limit
        D = 4       ## cho limit
        D = 2       ## numpy limit
    elif 0:
        N = 1000000
        D = 5       ## polyfit limit
        D = 4       ## cho limit
        D = 1       ## numpy limit

    def qpow(x, k):
        "x**k in quad precision"
        assert isinstance(k, int)
        if k < 0:
            x = p.div(p.one(), x)
            k = -k
        res = p.one()
        while k:
            if k & 1:
                res = p.mul(res, x)
            k >>= 1
            x   = p.mul(x, x)
        return res

    def pv(x):
        r"evaluate the model \pi + (x - N>>1) ** D"
        return p.add(
            p.to_quad(math.pi),
            qpow(p.sub(x, p.to_quad(N >> 1)), D)
        )

    def fac_(n):
        "factorial using recursion"
        if n < 2:
            return 1
        return n * fac(n - 1)

    FAC = { }
    def fac(n):
        "factorial"
        if n not in FAC:
            FAC[n] = fac_(n)
        return FAC[n]

    def bincof(n, k):
        "binomial coefficient"
        return fac(n) // fac(k) // fac(n - k)

    def itoq(n):
        "integer to quad"
        x = float(n)
        return (x, n - x)

    def exp_coef(k):
        "expected model coef about x=0"
        return \
            pow(N >> 1, k) * \
            bincof(D, k) * \
            (1, -1)[k & 1]

    ## expected coefs
    cv     = [itoq(exp_coef(k)) for k in range(D + 1)]
    cv[-1] = p.add(cv[-1], p.to_quad(math.pi))

    ## define the x and y values for the fit
    sc = 1.
    xv = [p.to_quad(x * sc) for x in range(N)]
    yv = [pv(x) for x in xv]
    wv = [p.one()] * len(xv)

    ## perform the fit
    plan = p.PolyfitPlan(D, xv, wv)
    fit  = plan.fit(yv)
    ev   = fit.evaluator()

    ## coefs about x=0
    print("polyfit cofs, x=0:")
    for i, c in enumerate(ev.coefs(xv[0], -1)):
        print("%2d %s" % (i, flist(c)))
    ## expected coefs about x=0
    print("expected:")
    for i, c in enumerate(cv):
        print("%2d %s" % (i, flist(c)))
    ## coefs about x=N>>1, expect [1, 0, ..., 0, pi]
    print("polyfit cofs, x=N>>1:")
    for i, c in enumerate(ev.coefs(p.to_quad(N >> 1), -1)):
        print("%2d %s" % (i, flist(c)))
    ## values at zero and N>>2, expect (N>>1)**D+pi and pi
    print("polyfit vals:")
    for i in (0, N >> 1):
        print("%6d %s" % (i, flist(ev(p.to_quad(i)))))

    ## numpy
    cofs = [p.to_quad(c) for c in chofit(xv, yv, wv, D)]
    print("numpy cofs, x=0:")
    for i, c in enumerate(cofs):
        print("%2d %s" % (i, flist(c)))

    def nv(x):
        x = p.to_quad(x)
        r = p.zero()
        for c in cofs:
            r = p.add(p.mul(r, x), c)
        return r

    print("numpy vals:")
    for i in (0, N >> 1):
        print("%6d %s" % (i, flist(nv(i))))

if __name__ == "__main__":
    demo()

## EOF
