#!/usr/bin/env python3

"""
pushing the envelope for polyfit and numpy
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
    for various numbers of data points
    """
    ## pylint: disable=too-many-locals

    ## "limit" means the estimation of pi
    ## is worse than 3.14159

    if 0:
        N = 10
        D = 10  ## polyfit and cho handle every degree
        D = 7  ## numpy limit
    elif 0:
        N = 100
        D = 16  ## polyfit limit
        D = 9  ## cho limit
        D = 4  ## numpy limit
    elif 0:
        N = 1000
        D = 10  ## polyfit limit
        D = 7  ## cho limit
        D = 3  ## numpy limit
    elif 1:
        N = 10000
        D = 7  ## polyfit limit
        D = 5  ## cho limit
        D = 2  ## numpy limit
    elif 0:
        N = 100000
        D = 6  ## polyfit limit
        D = 4  ## cho limit
        D = 2  ## numpy limit
    elif 0:
        N = 1000000
        D = 5  ## polyfit limit
        D = 4  ## cho limit
        D = 1  ## numpy limit

    def pv(x):
        r"evaluate the model \pi + (x - N>>1) ** D"
        return tl.p.add(
            tl.p.to_ddp(math.pi),
            tl.qx_to_the_k(tl.p.sub(x, tl.p.to_ddp(N >> 1)), D),
        )

    def exp_coef(k):
        "expected model coef about x=0"
        return pow(N >> 1, k) * tl.bincof(D, k) * (1, -1)[k & 1]

    ## expected coefs
    cv = [tl.itoq(exp_coef(k)) for k in range(D + 1)]
    cv[-1] = tl.p.add(cv[-1], tl.p.to_ddp(math.pi))

    ## define the x and y values for the fit
    sc = 1.0
    xv = tl.ddpvec(x * sc for x in range(N))
    yv = [pv(x) for x in xv]
    wv = [tl.p.one() for _ in xv]

    ## perform the fit
    plan = tl.p.PolyfitPlan(D, xv, wv)
    fit = plan.fit(yv)
    ev = fit.evaluator()

    ## coefs about x=0
    print("polyfit cofs, x=0:")
    for i, c in enumerate(ev.coefs(xv[0], -1)):
        print("%2d %s" % (i, tl.format_list(c)))
    ## expected coefs about x=0
    print("expected:")
    for i, c in enumerate(cv):
        print("%2d %s" % (i, tl.format_list(c)))
    ## coefs about x=N>>1, expect [1, 0, ..., 0, pi]
    print("polyfit cofs, x=N>>1:")
    for i, c in enumerate(ev.coefs(tl.p.to_ddp(N >> 1), -1)):
        print("%2d %s" % (i, tl.format_list(c)))
    ## values at zero and N>>2, expect (N>>1)**D+pi and pi
    print("polyfit vals:")
    for i in (0, N >> 1):
        print("%6d %s" % (i, tl.format_list(ev(tl.p.to_ddp(i)))))

    ## numpy
    cofs = chofit(xv, yv, wv, D)
    print("numpy cofs, x=0:")
    for i, c in enumerate(cofs):
        print("%2d %s" % (i, tl.format_list(c)))

    print("numpy vals:")
    for i in (0, N >> 1):
        print("%6d %s" % (i, tl.format_list(tl.ddpeval(i, cofs))))


if __name__ == "__main__":
    demo()

## EOF
