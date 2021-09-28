#!/usr/bin/env python3

"compute christoffel numbers from ortho polys"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import os
import sys
import time

sys.path.insert(0, "..")    ## pylint: disable=wrong-import-position

import polyfit as p

sys.stdout = os.fdopen(1, "w", 1)

def bis(    ## pylint: disable=too-many-arguments
        func, a, fa, b, fb, tol=1e-16
    ):
    "bisection root finder"
    assert p.to_float(p.mul(fa, fb)) < 0, (a, b, fa, fb)
    half = p.to_quad(0.5)
    for _ in range(100):
        delta = p.mul(half, p.sub(b, a))
        if abs(p.to_float(delta)) < tol:
            return a
        c  = p.add(a, delta)
        fc = func(c)
        if p.to_float(p.mul(fa, fc)) < 0:
            b, fb = c, fc
        elif not p.to_float(fc):
            return c
        else:
            a, fa = c, fc
    return c

def phi_k(x, k, plan):
    "compute orthogonal poly of deg k at x"
    b    = plan["b"]
    c    = plan["c"]
    x    = p.to_quad(x)
    pjm1 = p.zero()
    pj   = p.one()
    for j in range(k):
        pjp1 = p.sub(
            p.mul(p.sub(x, b[j]), pj),
            p.mul(c[j], pjm1)
        )
        pjm1 = pj
        pj   = pjp1
    return pj

def _roots(plan, k, x0, x1, ret):
    r"recursively compute the roots of \phi_k"
    ranges = [x0] + roots(plan, k - 1, x0, x1, ret) + [x1]
    rl     = [ ]
    for i in range(len(ranges) - 1):
        f  = lambda x: phi_k(x, k, plan)
        a  = ranges[i]
        fa = f(a)
        b  = ranges[i + 1]
        fb = f(b)
        rl.append(bis(f, a, fa, b, fb))
    return rl

def roots(plan, k, x0, x1, ret):
    r"return the roots of \phi_k"
    if k not in ret:
        ret[k] = _roots(plan, k, x0, x1, ret)
    return ret[k]

def allroots(plan):
    "return the roots of all of the phi_k keyed by k"
    ret = { 0: [ ] }
    x0  = p.to_quad(min(plan["x"]))
    x1  = p.to_quad(max(plan["x"]))
    roots(plan, plan["D"], x0, x1, ret)
    return ret

def _christoffel(plan, k, rl):
    r"""
    compute the christoffel numbers for gaussian summation:

    \sum_{k=1}^N w_k f(x_k) \approx \sum_{k=1}^D H_k f(r_k)

    r_k is the kth root of \phi_D
    H_k is the kth christoffel number

    the D-point gaussian summation formula is exact for
    polys f() up to degree 2D-1
    """
    rl  = rl[k]
    g   = plan["g"]
    ret = [ ]
    for r in rl:
        v = [ ]
        for j in range(k):
            pj = phi_k(r, j, plan)
            u  = p.div(p.mul(pj, pj), g[j + 1])
            p.vappend(v, u)
        ret.append(
            (
                r,
                p.div(p.one(), p.vectorsum(v))
            )
        )
    return ret

def christoffel(plan):
    r"compute christoffel numbers for all \phi_k"
    rl  = allroots(plan)
    ret = { }
    for d in range(0, plan["D"] + 1):
        ret[d] = _christoffel(plan, d, rl)
    return ret

def gauss_factory(plan):
    "return a factory to compute gaussian sums"
    Hs = christoffel(plan)
    def integrate(func, n):
        "weighted sum of function using n points"
        v = [ ]
        for x, H in Hs[n]:
            p.vappend(v, p.mul(H, func(x)))
        return p.vectorsum(v)
    def fintegrate(func, n):
        f = lambda x: p.to_quad(f(p.to_float(x)))
        return p.to_float(integrate(f, n))
    return Hs, integrate, fintegrate

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
    N    = 100000
    sc   = 1. #/ N
    xv   = [x * sc for x in range(N)]
    x0   = p.to_quad(min(xv))
    x1   = p.to_quad(max(xv))
    wv   = [1. for _ in xv]
    t0   = time.time()
    plan = p.polyfit_plan(D, xv, wv)
    print("plan  %.2e" % (time.time() - t0))

    def xk(x, k):
        x   = p.to_quad(x)
        ret = p.one()
        for _ in range(k):
            ret = p.mul(ret, x)
        return ret

    def check(l, k):
        "check the slow vs gaussian summation results"
        f   = lambda x: xk(x, k)
        v   = [ ]
        for w, x in zip(wv, xv):
            p.vappend(v, p.mul(p.to_quad(w), f(p.to_quad(x))))
        exp = p.to_float(p.vectorsum(v))
        obs = p.to_float(gauss(f, l))
        print(k, flist((exp, obs, obs - exp, abs(obs / exp - 1.))))

    t0 = time.time()
    Hs, gauss, fgauss = gauss_factory(plan)
    print("gauss %.2e" % (time.time() - t0))

    print()
    for l in range(1, D + 1):
        ## loop over gaussian summation order (#points)"
        print("order", l)
        print("H", flist(Hs[l], True))
        for k in range(2 * l):
            ## loop over x^k to show whether or not they agree
            check(l, k)
        print()

if __name__ == "__main__":
    demo()

## EOF
