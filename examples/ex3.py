#!/usr/bin/env python3

"example 3: accuracy benchmark: polyfit vs numpy using cpolyfit"

from __future__ import print_function

## pylint: disable=invalid-name,bad-whitespace

import array
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")

#from polyfit import Polyfit    ## pylint: disable=wrong-import-position
from cpolyfit import Polyfit    ## pylint: disable=wrong-import-position

sys.stdout = os.fdopen(1, "w", 1)

def peval(cofs, xv):
    """
    evaluate the polyomial given by cofs
    at each xv[] value. higher order terms
    come first in cofs
    """
    yv = array.array('d')
    for x in xv:
        r = 0.
        for c in cofs:
            r *= x
            r += c
        yv.append(r)
    return yv

def printfit(fit, xv, yv, dt, tag=""):
    ## pylint: disable=too-many-locals
    "print polyfit output"
    tag = (": " + tag) if tag else ""
    print("polyfit%s: dt %.4e" % (tag, dt))
    print("                  -max rel err-")
    print("deg     erms      erel     indx    coefs")
    errs = [fit.rms_err(i) for i in range(fit.maxdeg() + 1)]
    for i, err in enumerate(errs):
        cofs = fit.coefs(degree=i, x0=0.)
        maxrelerr, maxat = -1., None
        for j, x in enumerate(xv):
            exp = yv[j]
            obs = fit(x, degree=i)
            rel = abs(obs / exp - 1.)
            if rel > maxrelerr:
                maxrelerr, maxat = rel, j
        print(
            " %2d   %.1e   %.1e   %6d   %s" % \
            (i, err, maxrelerr, maxat, cofs)
        )
    print()

def do_polyfit(cofs, xv, wv=None):
    "compute and print polyfit"
    yv = peval(cofs, xv)
    if (wv is None) or (isinstance(wv, str) and wv == "equal"):
        tag = "unity weights"
        wv  = array.array('d', [1.] * len(xv))
    elif isinstance(wv, str) and wv == "minrel":
        tag = "minrel weights"
        wv  = array.array('d', [y**-2. for y in yv])
    else:
        tag = "custon weights"
    t0  = time.time()
    fit = Polyfit(len(cofs) - 1, xv, yv, wv)
    dt  = time.time() - t0
    printfit(fit, xv, yv, dt, tag)

def npfit(xv, yv, wv, D):
    "numpy fit"
    xv = np.array(xv)
    yv = np.array(yv)
    wv = np.array(wv)
    xa = wv
    xp = [xa]
    for i in range(D * 2):
        xa = xa * xv
        xp.append(xa)
    b = [ ]
    for i in range(D):
        b.append(np.dot(yv, xp[i]))
    b  = np.array(b)
    mx = [ ]
    for v in xp:
        mx.append(np.sum(v))
    mx = np.array(mx)
    a  = [ ]
    for i in range(D):
        a.append(mx[i:i+D])
    a = np.array(a)

    cofs = list(np.linalg.solve(a, b))
    return cofs

def do_numpy(cofs, xv, wv=None):
    ## pylint: disable=too-many-locals
    "compute and print npfit"
    yv = peval(cofs, xv)
    if (wv is None) or (isinstance(wv, str) and wv == "equal"):
        tag = "equal weights"
        wv  = [1.] * len(xv)
    elif isinstance(wv, str) and wv == "minrel":
        tag = "minrel weights"
        wv  = [y**-2. for y in yv]
    else:
        tag = "custon weights"
    t0  = time.time()
    fit = npfit(xv, yv, wv, D=len(cofs))
    dt  = time.time() - t0
    tag = (": " + tag) if tag else ""
    print("numpy%s: dt %.4e" % (tag, dt))
    print("                  -max rel err-")
    print("deg     erms      erel     indx    coefs")
    cofs  = fit
    ceval = cofs[:]
    ceval.reverse()
    def pv(x):
        r = 0.
        for c in ceval:
            r *= x
            r += c
        return r
    pred = [pv(x) for x in xv]
    rms  = sum((p - o)**2 for p, o in zip(pred, yv))
    rms  = math.sqrt(rms / len(xv))
    maxrelerr, maxat = -1., 0
    for j, x in enumerate(xv):
        exp = yv[j]
        obs = pv(x)
        rel = abs(obs / exp - 1.)
        if rel > maxrelerr:
            maxrelerr, maxat = rel, j
    print(
        " %2d   %.1e   %.1e   %6d   %s" % \
        (len(cofs)-1, rms, maxrelerr, maxat, cofs)
    )
    print()

def do_both(cofs, xv, wv=None):
    "do polyfit and numpy"
    xv = array.array('d', xv)
    do_polyfit(cofs, xv, wv)
    do_numpy(cofs, xv, wv)

print("#" * 72)
print("UNSCALED-X EQUAL WEIGHT CUBIC FIT")
do_both([2, 1, -1, math.pi], range(100000), "equal")

print("#" * 72)
print("SCALED-X EQUAL WEIGHT CUBIC FIT")
do_both([2, 1, -1, math.pi], [x * 1e-5 for x in range(100000)], "equal")

print("#" * 72)
print("UNSCALED-X EQUAL WEIGHT QUARTIC FIT")
do_both([0, 2, 1, -1, math.pi], range(100000), "equal")

print("#" * 72)
print("SCALED-X EQUAL WEIGHT QUARTIC FIT")
do_both([0, 2, 1, -1, math.pi], [x * 1e-5 for x in range(100000)], "equal")

print("#" * 72)
print("UNSCALED-X RELATIVE WEIGHT CUBIC FIT")
do_both([2, 1, -1, math.pi], range(100000), "minrel")

print("#" * 72)
print("SCALED-X RELATIVE WEIGHT CUBIC FIT")
do_both([2, 1, -1, math.pi], [x * 1e-5 for x in range(100000)], "minrel")

print("#" * 72)
print("UNSCALED-X RELATIVE WEIGHT QUARTIC FIT")
do_both([0, 2, 1, -1, math.pi], range(100000), "minrel")

print("#" * 72)
print("SCALED-X RELATIVE WEIGHT QUARTIC FIT")
do_both([0, 2, 1, -1, math.pi], [x * 1e-5 for x in range(100000)], "minrel")

print("#" * 72)
print("UNSCALED-X EQUAL WEIGHT 10TH DEGREE FIT")
do_both([0, 0, 0, 0, 0, 0, 0, 2, 1, -1, math.pi], range(100000), "equal")

print("#" * 72)
print("SCALED-X EQUAL WEIGHT 10TH DEGREE FIT")
do_both([0, 0, 0, 0, 0, 0, 0, 2, 1, -1, math.pi], [x * 1e-5 for x in range(100000)], "equal")

print("#" * 72)
print("UNSCALED-X RELATIVE WEIGHT 10TH DEGREE FIT")
do_both([0, 0, 0, 0, 0, 0, 0, 2, 1, -1, math.pi], range(100000), "minrel")

print("#" * 72)
print("SCALED-X RELATIVE WEIGHT 10TH DEGREE FIT")
do_both([0, 0, 0, 0, 0, 0, 0, 2, 1, -1, math.pi], [x * 1e-5 for x in range(100000)], "minrel")
