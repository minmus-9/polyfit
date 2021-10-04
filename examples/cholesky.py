#!/usr/bin/env python

"""
cholesky decomp and solver for polyfit comparisons

the decomp is adapted from

    https://github.com/Verolop/cholesky_decomposition.git
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

import sys

sys.path.insert(0, "..")

from polyfit import (   ## pylint: disable=wrong-import-position
    zero, one, div, mul, sub, sqrt,
    vappend, vectorsum, to_quad
)

def cholesky_decomposition(A):
    """
    perform the decomposition

    this doesn't do pivoting, etc, but it sure is simple
    """
    n   = len(A)

    uno = one()
    zed = zero()

    A   = [[to_quad(z) for z in row] for row in A]
    L   = [[zed] * n for i in range(n)]

    for i in range(n):
        for k in range(i+1):
            v = [ ]
            for j in range(k):
                vappend(v, mul(L[i][j], L[k][j]))
            suma = vectorsum(v)

            if i == k:
                L[i][k] = sqrt(sub(to_quad(A[i][i]), suma))
            else:
                L[i][k] = mul(
                    div(uno, L[k][k]),
                    sub(A[i][k], suma)
                )
    return L

def cholesky_forward_sub(L, b):
    "fwd sub on a lower triangular matrix"
    y = [to_quad(z) for z in b]
    n = len(L)
    for i in range(n):
        for j in range(i):
            y[i] = sub(y[i], mul(L[i][j], y[j]))
        y[i] = div(y[i], L[i][i])
    return y

def cholesky_back_sub(L, y):
    "back sub on transpose of a lower triangular matrix"
    x = [to_quad(z) for z in y]
    n = len(L)
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            ## L[j][i] is L transpose
            x[i] = sub(x[i], mul(L[j][i], x[j]))
        x[i] = div(x[i], L[i][i])
    return x

def cholesky_solve(L, b):
    "given a decomp L of A, solve A x = b"
    return cholesky_back_sub(L, cholesky_forward_sub(L, b))

def cholesky(A, b):
    "solve A x = b using the cholesky decomp"
    L = cholesky_decomposition(A)
    return cholesky_solve(L, b)

def chofit(xv, yv, wv, D):
    "cholesky least-squares fit, quad-precision"
    ## pylint: disable=too-many-locals
    xv = [to_quad(x) for x in xv]
    yv = [to_quad(y) for y in yv]
    wv = [to_quad(w) for w in wv]
    xa = wv[:]          ## accumulator
    mx = [ ]            ## quad-prec moments
    b  = [ ]            ## quad-prec rhs in Ac=r
    for i in range((D + 1) * 2):
        if i <= D:
            ## compute rhs
            v = [ ]
            for x, y in zip(xa, yv):
                vappend(v, mul(x, y))
            b.append(vectorsum(v))
        ## compute moments up to 2D+1
        v = [ ]
        for x in xa:
            vappend(v, x)
        mx.append(vectorsum(v))
        for j, x in enumerate(xa):
            xa[j] = mul(x, xv[j])
    A = [ ]
    for i in range(D + 1):
        A.append(mx[i:i+D+1])
    cofs = cholesky(A, b)
    ## get 'em ready for horner's method
    cofs.reverse()
    return cofs

def test():
    "test code"
    A = [
        [ 1,  3,  5],
        [ 3, 13, 23],
        [ 5, 23, 42]
    ]
    b = [2, 2, 4]
    from polyfit import to_float    ## pylint: disable=import-outside-toplevel
    assert [to_float(z) for z in cholesky(A, b)] == [7, -5, 2]
    print("pass")

if __name__ == "__main__":
    test()

## EOF
