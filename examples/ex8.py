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

## pylint: disable=invalid-name,bad-whitespace,consider-using-f-string

import time

import testlib as tl


def demo():
    "demo code"
    ## pylint: disable=too-many-locals

    D = 8
    N = 10000
    sc = 1.0  # / N
    xv = [x * sc for x in range(N)]
    wv = [1.0 for _ in xv]

    t0 = time.time()
    plan = tl.p.PolyfitPlan(D, xv, wv)
    print("plan %.2e" % (time.time() - t0))

    def check(l, k):
        "check the slow vs gaussian quadrature results"

        def f(x):
            return tl.qx_to_the_k(x, k)

        ## compute the large sum
        v = []
        for w, x in zip(wv, xv):
            tl.p.vappend(v, tl.p.mul(tl.p.to_ddp(w), f(x)))
        exp = tl.p.vectorsum(v)
        ## do the quadrature
        obs = quad.ddpquad(f, l)
        ## compute difference and rel error in DDP
        dif = tl.p.sub(obs, exp)
        rel = tl.p.div(dif, exp)
        ## then dumb everything down to double for printing
        rel = tl.p.to_float(rel)
        dif = tl.p.to_float(dif)
        exp = tl.p.to_float(exp)
        obs = tl.p.to_float(obs)
        print(k, tl.format_list((exp, obs, dif, rel)))

    t0 = time.time()
    quad = tl.q.PolyplusQuadrature(plan)
    print("quad %.2e" % (time.time() - t0))

    ## quick serialization test
    ser = quad.to_data()
    assert isinstance(ser, dict)
    quad = tl.q.PolyplusQuadrature.from_data(ser)

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
