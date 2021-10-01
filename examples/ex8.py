#!/usr/bin/env python3

"""
gaussian quadrature demo

if you want to compute

    sum(f(x_i) * w_i for x_i, w_i in zip(xv, wv))

you can replace this with gaussian quadrature as

    sum(f(z_i) * H_i for z_i, H_i in zip(Z, H))

the difference is that D = len(Z) is much smaller
than len(xv). Z and H are generated from a fit
plan for xv and wv. this module provides a function
to accurately integrate f().

the quadrature formula is exact for polynomial
functions f() up to degree 2D-1.

for non-polynomial functions, the error is
proportional to the 2D-th derivative of f()
divided by (2D)!
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import os
import sys
import time

sys.path.insert(0, "..")    ## pylint: disable=wrong-import-position

import polyfit  as p
import polyplus as q

sys.stdout = os.fdopen(1, "w", 1)

def demo():
    "demo code"
    ## pylint: disable=too-many-locals

    def flist(l, lf=False):
        "format items as high-precision"
        f1 = "%23.16e"
        f2 = "(%23.16e, %23.16e)"
        ll = [ ]
        s  = "\n  " if lf else " "
        for x in l:
            if isinstance(x, tuple):
                ll.append("(%s, %s)" % (f2 % x[0], f2 % x[1]))
            else:
                ll.append(f1 % x)
        return s + s.join(ll)

    D    = 8
    N    = 10000
    sc   = 1. #/ N
    xv   = [x * sc for x in range(N)]
    wv   = [1. for _ in xv]

    t0   = time.time()
    plan = p.PolyfitPlan(D, xv, wv)
    print("plan %.2e" % (time.time() - t0))

    def xk(x, k):
        "compute x**k in quad precision"
        x   = p.to_quad(x)
        ret = p.one()
        while k:
            if k & 1:
                ret = p.mul(ret, x)
            k >>= 1
            x   = p.mul(x, x)
        return ret

    def check(l, k):
        "check the slow vs gaussian quadrature results"
        f   = lambda x: xk(x, k)
        ## compute the large sum
        v   = [ ]
        for w, x in zip(wv, xv):
            p.vappend(v, p.mul(p.to_quad(w), f(p.to_quad(x))))
        exp = p.vectorsum(v)
        ## do the quadrature
        obs = quad.qquad(f, l)
        ## compute difference and rel error in quad prec
        dif = p.sub(obs, exp)
        rel = p.div(dif, exp)
        ## then dumb everything down to double for printing
        rel = p.to_float(rel)
        dif = p.to_float(dif)
        exp = p.to_float(exp)
        obs = p.to_float(obs)
        print(k, flist((exp, obs, dif, rel)))

    t0 = time.time()
    quad = q.PolyplusQuadrature(plan)
    print("quad %.2e" % (time.time() - t0))

    ## quick serialization test
    ser  = quad.to_data()
    assert isinstance(ser, dict)
    quad = q.PolyplusQuadrature.from_data(ser)

    print()
    for l in range(1, D + 1):
        ## loop over gaussian quadrature order (#points)"
        print("order", l)
        for k in range(2 * l):
            ## loop over x^k and show whether or not they agree
            check(l, k)
        print()

if __name__ == "__main__":
    demo()

## EOF
