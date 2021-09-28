#!/usr/bin/env python3

"integral of fit poly"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import sys

sys.path.insert(0, "..")
import polyfit as p ## pylint: disable=wrong-import-position

def make_integrator_factory(xv, yv, wv, D):
    "gaussian summation factory"
    plan = p.polyfit_plan(D, xv, wv)
    fit  = p.polyfit_qfit(plan, yv)
    def factory(deg):
        "return coefs and integrators"
        coefs = p.polyfit_qcoefs(plan, fit, x0=0., deg=deg)
        for j in range(deg, -1, -1):
            i = deg - j
            fac = p.div(p.one(), p.to_quad(j + 1))
            coefs[i] = p.mul(coefs[i], fac)
        coefs.append(p.zero())
        def integrate(x):
            "quad precision integrator"
            ret = p.zero()
            for c in coefs:
                ret = p.add(p.mul(ret, x), c)
            return ret
        def integratef(x):
            "double precision integrator"
            return p.to_float(integrate(p.to_quad(x)))
        return coefs, integrate, integratef
    return factory

def demo():
    "demo code"
    cv = [3, 2, 1]
    def pv(x):
        r = 0.
        for c in cv:
            r *= x
            r += c
        return r
    D  = 2
    N  = 4
    sc = 1.
    xv = [x * sc for x in range(N)]
    yv = [pv(x) for x in xv]
    wv = [1. for _ in xv]

    factory = make_integrator_factory(xv, yv, wv, D)
    coefs, integ, integf = factory(D)   ## pylint: disable=unused-variable

    print([integf(x) for x in xv])
    ## should be [0, 3, 14, 39]

if __name__ == "__main__":
    demo()

## EOF
