#!/usr/bin/env python3

"""
pushing the envelope for polyfit and numpy: extrapolation
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math
import sys

sys.path.insert(0, "..")

import polyfit as p     ## pylint: disable=wrong-import-position

from np import npfit    ## pylint: disable=wrong-import-position

def flist(l):
    "format a list to 16 sigfigs"
    if not isinstance(l, (list, tuple)):
        l = [l]
    return " ".join("%24.15e" % x for x in l)

def demo():
    # pylint: disable=too-many-statements,using-constant-test
    """
    this demo shows the limits of polyfit and numpy
    for various numbers of data points
    """
    ## pylint: disable=unnecessary-comprehension,too-many-locals

    N = 1000
    D = 3

    ## here "limit" means using the largest X such that
    ## the model and fit(X) match to 6 sigfigs

    X = p.to_quad(1e102)    ## polyfit never craps out
    X = p.to_quad(2e2)      ## limit for numpy

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

    EXP = pv(X)

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
    sc = 1 / N
    xv = [p.to_quad(x * sc) for x in range(N)]
    yv = [pv(x) for x in xv]
    wv = [p.one()] * len(xv)

    ## perform the fit
    plan = p.PolyfitPlan(D, xv, wv)
    fit  = plan.fit(yv)
    ev   = fit.evaluator()

    ## values at zero and N>>2, expect (N>>1)**D+pi and pi
    print("polyfit vals:")
    for i in (0, N >> 1, X):
        print("%14.5e %s" % (p.to_float(i), flist(ev(p.to_quad(i)))))
    print("exp            %s" % flist(EXP))

    ## numpy
    cofs = npfit(xv, yv, wv, D)

    def nv(x):
        x = p.to_float(x)
        r = 0.
        for c in cofs:
            r *= x
            r += c
        return r

    print("numpy vals:")
    for i in (0, N >> 1, X):
        print("%14.5e %s" % (p.to_float(i), flist(nv(i))))
    print("exp            %s" % flist(EXP))

if __name__ == "__main__":
    demo()

## EOF
