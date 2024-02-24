#!/usr/bin/env python3

"""
relative error in coefs, polyfit vs numpy
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import math

import testlib as tl  ## pylint: disable=wrong-import-position

# chofit = tl.n.chofit
chofit = tl.c.chofit


def demo():
    "compute rel err in poly coefs"
    ## pylint: disable=too-many-locals

    ## poly coefficients to fit, highest degree first
    cv = [2, 1, -1, math.pi]
    cvee = tl.ddpvec(cv)

    ## define the x and y values for the fit
    N = 10000
    xv = tl.ddpvec(range(N))
    yv = [tl.ddpeval(x, cvee) for x in xv]

    ## weights:
    ##     uniform to minimize the max residual
    wv = [tl.p.one() for _ in xv]

    ## perform the fit
    D = len(cv) - 1
    plan = tl.p.PolyfitPlan(D, xv, wv)
    fit = plan.fit(yv)
    ev = fit.evaluator()

    exp = tl.ddpeval(tl.p.to_ddp(N >> 1), cvee)

    ## print rel errs in coefs
    obs = [tl.p.to_float(c) for c in ev.coefs(xv[0], -1)]
    rel = [abs(o / e - 1.0) if e else o for o, e in zip(obs, cv)]
    print("polyfit:")
    print("  coefs ", tl.format_list(obs))
    print("  relerr", tl.format_list(rel))
    print("    P(0)", tl.format_list(ev(tl.p.zero())))
    print(" P(N>>1)", tl.format_list(ev(tl.p.to_ddp(N >> 1))))
    print("     exp", tl.format_list(exp))

    ## numpy
    cof = chofit(xv, yv, wv, D)

    obs = [tl.p.to_float(c) for c in cof]
    rel = [abs(o / e - 1.0) if e else o for o, e in zip(obs, cv)]
    print("numpy:")
    print("  coefs ", tl.format_list(obs))
    print("  relerr", tl.format_list(rel))
    print("    P(0)", tl.format_list(tl.ddpeval(tl.p.zero(), cof)))
    print(" P(N>>1)", tl.format_list(tl.ddpeval(tl.p.to_ddp(N >> 1), cof)))
    print("     exp", tl.format_list(exp))


if __name__ == "__main__":
    demo()

## EOF
