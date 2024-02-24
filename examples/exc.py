#!/usr/bin/env python3

"""
pushing the envelope for polyfit and numpy: extrapolation
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace,consider-using-f-string

import math

import testlib as tl

# chofit = tl.n.chofit
chofit = tl.c.chofit


def demo():
    # pylint: disable=too-many-statements,using-constant-test
    """
    this demo shows the limits of polyfit and numpy
    extrapolation outside the fit interval
    """
    ## pylint: disable=too-many-locals

    N = 10000
    D = 3

    ## here "limit" means using the largest X such that
    ## the model and fit(X) match to 6 sigfigs

    X = tl.p.to_ddp(1e10)  ## polyfit never craps out
    X = tl.p.to_ddp(2e2)  ## limit for numpy

    def pv(x):
        r"evaluate the model \pi + (x - N>>1) ** D"
        return tl.p.add(
            tl.p.to_ddp(math.pi),
            tl.qx_to_the_k(tl.p.sub(x, tl.p.to_ddp(N >> 1)), D),
        )

    EXP = pv(X)

    def exp_coef(k):
        "expected model coef about x=0"
        return pow(N >> 1, k) * tl.bincof(D, k) * (1, -1)[k & 1]

    ## expected coefs
    cv = [tl.itoq(exp_coef(k)) for k in range(D + 1)]
    cv[-1] = tl.p.add(cv[-1], tl.p.to_ddp(math.pi))

    ## define the x and y values for the fit
    sc = 1 / N
    xv = [tl.p.to_ddp(x * sc) for x in range(N)]
    yv = [pv(x) for x in xv]
    wv = [tl.p.one()] * len(xv)

    ## perform the fit
    plan = tl.p.PolyfitPlan(D, xv, wv)
    fit = plan.fit(yv)
    ev = fit.evaluator()

    ## values at zero and N>>2, expect (N>>1)**D+pi and pi
    print("polyfit vals:")
    for i in (0, N >> 1, X):
        print(
            "%14.5e %s"
            % (tl.p.to_float(i), tl.format_list(ev(tl.p.to_ddp(i))))
        )
    print("exp            %s" % tl.format_list(EXP))

    ## numpy
    cofs = chofit(xv, yv, wv, D)

    print("numpy vals:")
    for i in (0, N >> 1, X):
        print(
            "%14.5e %s"
            % (tl.p.to_float(i), tl.format_list(tl.ddpeval(i, cofs)))
        )
    print("exp            %s" % tl.format_list(EXP))


if __name__ == "__main__":
    demo()

## EOF
