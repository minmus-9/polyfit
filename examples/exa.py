#!/usr/bin/env python3

"example usage of polyfit serialization"

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import json
import math

import testlib as tl

def demo():
    "demo of the api"
    ## pylint: disable=too-many-locals

    ## poly coefficients to fit, highest degree first
    cv   = [2, 1, -1, math.pi]
    cvee = tl.qvec(cv)

    ## define the x and y values for the fit
    N  = 10000
    xv = tl.qvec(range(N))
    yv = [tl.qeval(x, cvee) for x in xv]

    ## weights:
    ##     uniform to minimize the max residual
    wv = [tl.p.one() for _ in xv]

    ## perform the fit
    D    = len(cv) - 1
    plan = tl.p.PolyfitPlan(D, xv, wv)
    fit1 = plan.fit(yv)
    ev1  = fit1.evaluator()

    ## serialize and deserialize
    data = json.dumps(fit1.to_data())
    print("json   %d bytes" % len(data))
    print()
    fit2 = tl.p.PolyfitFit.from_data(json.loads(data))
    ev2  = fit2.evaluator()

    idx = N >> 1
    print("value1 %.1f %s" % (tl.p.to_float(xv[idx]), tl.format_list(ev1(xv[idx], nder=0))))
    print("value2 %.1f %s" % (tl.p.to_float(xv[idx]), tl.format_list(ev2(xv[idx], nder=0))))
    print()

    ## compare coefs
    x = "  ".join(tl.format_list(q) for q in ev1.coefs(xv[0], D))
    print("coefs  ", x)
    x = "  ".join(tl.format_list(q) for q in ev2.coefs(xv[0], D))
    print("coefs  ", x)

if __name__ == "__main__":
    demo()

## EOF
