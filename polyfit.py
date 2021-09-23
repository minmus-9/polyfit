#!/usr/bin/env python3

"quad precision orthogonal polynomial least squares fitting"

from __future__ import print_function

## pylint: disable=invalid-name,bad-whitespace
## pylint: disable=useless-object-inheritance
## pylint: disable=unnecessary-comprehension
## XXX pylint: disable=missing-docstring

import math

## {{{ quad precision routines from ogita et al
def twosum(a, b):
    "6 flops, algorithm 3.1 from ogita"
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y

def twodiff(a, b):
    "6 flops, subtraction version of twosum()"
    x = a - b
    z = x - a
    y = (a - (x - z)) - (b + z)
    return x, y

def split(a, FACTOR = 1. + 2. ** 27):
    "4 flops, algorithm 3.2 from ogita"
    c = FACTOR * a
    x = c - (c - a)
    y = a - x
    return x, y

def twoproduct(a, b):
    "23 flops, algorithm 3.3 from ogita"
    x      = a * b
    a1, a2 = split(a)
    b1, b2 = split(b)
    y      = a2 * b2 - (x - a1 * b1 - a2 * b1 - a1 * b2)
    return twosum(x, y)

def sum2s(p):
    "7n-1 flops, algorithm 4.1 from ogita"
    pi, sigma = p[0], 0.
    for i in range(1, len(p)):
        pi, q  = twosum(pi, p[i])
        sigma += q
    return twosum(pi, sigma)

def vsum(p):
    "6(n-1) flops, algorithm 4.3 from ogita"
    im1 = 0
    for i in range(1, len(p)):
        p[i], p[im1] = twosum(p[i], p[im1])
        im1 = i
    return p

def sumkcore(p, K):
    "6(K-1)(n-1) flops, algorithm 4.8 from ogita"
    for _ in range(K - 1):
        p = vsum(p)
    return p

def sumk(p, K):
    "(6K+1)(n-1)+6 flops, algorithm 4.8 from ogita"
    p = sumkcore(p, K)
    return sum2s(p)

def vectorsum(vec):
    "19n-13 flops, sumk() with K=3"
    return sumk(vec, K=3)
## }}}
## {{{ utility functions
def zero():
    return (0., 0.)

def one():
    return (1., 0.)

def vappend(vec, x):
    "append quad to vector"
    vec.extend(x)

def to_quad(x):
    "float to quad"
    return x if isinstance(x, tuple) else (float(x), 0.)

def to_float(x):
    "quad to float"
    return x[0] if isinstance(x, tuple) else float(x)
## }}}
## {{{ quad precision arithmetic
def add(x, y):
    "14 flops"
    x, xx = x
    y, yy = y
    z, zz = twosum(x, y)
    return twosum(z, zz + xx + yy)

def sub(x, y):
    "14 flops"
    x, xx = x
    y, yy = y
    z, zz = twodiff(x, y)
    return twosum(z, zz + xx - yy)

def mul(x, y):
    "33 flops"
    x, xx = x
    y, yy = y
    z, zz = twoproduct(x, y)
    zz   += xx * y + x * yy
    return twosum(z, zz)

def div(x, y):
    "36 flops, from dekker"
    x, xx = x
    y, yy = y
    c     = x / y
    u, uu = twoproduct(c, y)
    cc    = (x - u - uu + xx - c * yy) / y
    return twosum(c, cc)

def sqrt(x):
    "35 flops, from dekker"
    x, xx = x
    if not (x or xx):
        return zero()
    c     = math.sqrt(x)
    u, uu = twoproduct(c, c)
    cc    = (x - u - uu + xx) * 0.5 / c
    return twosum(c, cc)
## }}}
## {{{ polyfit_plan
def polyfit_plan(maxdeg, xv, wv):
    """
    given x values in xv[] and positive weights in wv[],
    make a plan to perform least squares fitting up to
    degree maxdeg.

    returns a plan object than can be json-(de)serialized.

    this is code for "compute everything need to calculate
    an expansion in xv- and wv-specific orthogonal
    polynomials".
    """
    ## pylint: disable=too-many-locals

    ## convert to quad
    xv = [to_quad(x) for x in xv]
    wv = [to_quad(w) for w in wv]
    N  = len(xv)
    b  = [ ]            ## recurrence coefs b_k
    c  = [ ]            ## recurrence coefs c_k
    g  = [one()]        ## \gamma_k^2 \equiv (\phi_k, \phi_k)
    r  = {
        "D": maxdeg,
        "N": N,
        "b": b,
        "c": c,
        "g": g,
        "x": xv,        ## needed for actual fit
        "w": wv         ## needed for actual fit
    }
    ## \phi_{k-1} and \phi_k
    phi_km1 = [zero()] * N ## \phi_{-1}
    phi_k   = [one()]  * N ## \phi_0

    for k in range(maxdeg + 1):
        bvec, gvec = [ ], [ ]
        for i in range(N):
            p   = phi_k[i]
            ## w_i \phi_k^2(x_i)
            wp2 = mul(wv[i], mul(p, p))
            ## w_i x_i \phi_k^2(x_i)
            vappend(bvec, mul(xv[i], wp2))
            ## w_i \phi_k^2(x_i)
            vappend(gvec, wp2)
        ## compute g_k = (\phi_k, \phi_k), b_k, and c_k
        gk = vectorsum(gvec)
        bk = div(vectorsum(bvec), gk)
        ck = div(gk, g[k])
        g.append(gk)
        b.append(bk)
        c.append(ck)
        ## if we aren't done, update pk[] and pkm1[] for next round
        if k == maxdeg:
            break
        for i in range(N):
            ## \phi_{k+1}(x_i) = (x_i - b_k) \phi_k(x_i) -
            ##                     c_k \phi_{k-1}(x_i)
            phi_kp1 = sub(
                mul(sub(xv[i], bk), phi_k[i]),
                mul(ck, phi_km1[i])
            )
            ## rotate the polys
            phi_km1[i] = phi_k[i]
            phi_k[i]   = phi_kp1
    c.append(zero())    ## needed in polyfit_eval
    return r
## }}}
## {{{ polyfit_fit
def polyfit_fit(plan, yv):
    """
    given a previously generated plan and a set of y values
    in yv[], compute all least squares fits to yv[] up to
    degree maxdeg.

    returns (rms_errors, evaluator, coef_evaluator) where
    rms_errors is a vector of rms fit errors for each possible
    degree, evaluator is a function to evaluate the fit
    polynomial, and coef_evaluator is a function to generate
    polynomial coefficients for the standard x_k basis.
    """
    ## pylint: disable=too-many-locals
    N, D = plan["N"], plan["D"]
    b, c = plan["b"], plan["c"]
    g    = plan["g"]
    wv   = plan["w"]
    xv   = plan["x"]

    a, e = [ ], [ ]     ## fit coefs and rms errors
    rv   = [to_quad(y) for y in yv] ## residuals

    ## \phi_{k-1} and \phi_k
    phi_km1 = [zero()] * N
    phi_k   = [one()]  * N
    for k in range(D + 1):
        ## compute ak as (residual, \phi_k) / (\phi_k, \phi_k)
        avec = [ ]
        for i in range(N):
            vappend(avec, mul(wv[i], mul(rv[i], phi_k[i])))
        ak = div(vectorsum(avec), g[k + 1])
        a.append(ak)

        ## remove the \phi_k component from the residual
        ## compute rms error for this degree
        evec = [ ]
        for i in range(N):
            rv[i] = r = sub(rv[i], mul(ak, phi_k[i]))
            vappend(evec, mul(r, r))
        e.append(sqrt(div(vectorsum(evec), to_quad(N))))


        ## if we aren't done, update pk[] and pkm1[] for next round
        if k == D:
            break
        for i in range(N):
            ## \phi_{k+1}(x_i) = (x_i - b_k) \phi_k(x_i) -
            ##                     c_k \phi_{k-1}(x_i)
            phi_kp1 = sub(
                mul(sub(xv[i], b[k]), phi_k[i]),
                mul(c[k], phi_km1[i])
            )
            ## rotate the polys
            phi_km1[i] = phi_k[i]
            phi_k[i]   = phi_kp1
    ## return rms errors by degree, a poly evaluator, and a coef evaluator
    return (
        [to_float(err) for err in e],
        (lambda x, deg=-1, nder=-1: polyfit_eval(plan, a, x, deg, nder)),
        (lambda x, deg=-1: polyfit_coefs(plan, a, x, deg))
    )
## }}}
## {{{ polyfit_eval
def polyfit_eval_(plan, a, x, deg=-1, nder=-1):
    """
    given a plan, a set of expansion coefficients generated
    by polyfit_fit, a point x, a least squares fit degree
    deg, and a desired number of derivatives to compute
    nder, calculate and return the value of the polynomial
    and any requested derivatives.

    if deg is negative, use maxdeg instead. if nder is
    negative, use the final value of deg; otherwise, compute
    ndeg derivatives of the least squares polynomial of
    degree deg.

    returns a list whose first element is the value of the
    least squares polynomial of degree deg at x. subsequent
    elements are the requested derivatives.
    """
    ## pylint: disable=too-many-locals
    b, c, D = plan["b"], plan["c"], plan["D"]

    if deg < 0:
        deg = D
    if nder < 0:
        nder = deg

    ## z_k^{(j-1)} and z_k^{(j)} for clenshaw's recurrence
    zjm1 = a[:deg+1] + [zero(), zero()] ## init to a_kj
    zj   = [zero()] * (deg + 3)

    fac  = one()            ## j! factor
    lim  = min(deg, nder)   ## max degree to compute
    x    = to_quad(x)
    ret  = [ ]              ## return value
    for j in range(lim + 1):
        if j > 1:
            fac = mul(fac, to_quad(j))
        ## compute z_j^{(j)} using the recurrence
        for k in range(deg, j - 1, -1):
            t = k - j
            ## z_k^{(j)} = z_k^{(j-1)} +
            ##              (x - b_t) z_{k+1}^{(j)} -
            ##              c_{t+1} z_{k+2}^{(j)}
            tmp   = sub(
                mul(sub(x, b[t]), zj[k + 1]),
                mul(c[t + 1], zj[k + 2])
            )
            zj[k] = add(zjm1[k], tmp)
        ## save j! z_j^{(j)}
        ret.append(mul(fac, zj[j]))
        ## update z if we aren't done
        if j == lim:
            break
        ## update zjm1
        zjm1[:] = zj
        ## zj only needs last 2 elements cleared
        zj[-2:] = [zero(), zero()]
    ## returns quad precision (for polyfit_coefs)
    return ret

def polyfit_eval(plan, a, x, deg=-1, nder=-1):
    ## get float values
    r = polyfit_eval_(plan, a, x, deg, nder)
    r = [to_float(v) for v in r]
    ## return scalar if no derivs
    return r[0] if len(r) == 1 else r
## }}}
## {{{ polyfit_coefs
def polyfit_coefs(plan, a, x0=0., deg=-1):
    """
    given a plan, a set of expansion coefficients generated
    by polyfit_fit, a center point x0, and a least squares
    fit degree, return the coefficients of powers of (x - x0)
    with the highest powers first. if deg is negative (the
    default), use maxdeg instead.
    """
    ## get value and derivs, divide by j!
    vals = polyfit_eval_(plan, a, x0, deg, deg)
    fac  = one()
    for j in range(2, len(vals)):
        fac     = div(fac, to_quad(j))
        vals[j] = mul(vals[j], fac)
    ## get highest power first and convert to float
    vals.reverse()
    return [to_float(v) for v in vals]
## }}}
## {{{ polyfit_maxdeg and polyfit_npoints
def polyfit_maxdeg(plan):
    "return the maximum possible fit degree"
    return plan["D"]

def polyfit_npoints(plan):
    "return the number of data points being fit"
    return plan["N"]
## }}}
## {{{ Polyfit classes
class PolyfitFit(object):
    """
    polynomial fitter returned by PolyfitPlan.fit()
    """

    def __init__(self, plan, yv):
        """
        given a plan and a set of y values yv[], call polyfit_plan
        to compute the least squares fits up to degree maxdeg.
        """
        self.plan = plan
        self.rms, self.eval, self.cofs = polyfit_fit(plan, yv)

    def __call__(self, x, deg=-1, nder=-1):
        return self.eval(x, deg, nder)

    def coefs(self, x0, deg=-1):
        """
        return the coefficients of the fit polynomial of degree
        deg about (x - x0). if deg is negative, use maxdeg
        instead.
        """
        return self.cofs(x0, deg)

    def rms_errors(self):
        """
        return a list of rms errors, one per fit degree. use them to
        detect overfitting.
        """
        return self.rms

class PolyfitPlan(object):
    ## pylint: disable=too-few-public-methods

    def __init__(self, maxdeg, xv, wv):
        self.plan = polyfit_plan(maxdeg, xv, wv)

    def fit(self, yv):
        return PolyfitFit(self.plan, yv)
## }}}
def demo():
    ## pylint: disable=too-many-locals
    cv = [1, -2, 1]
    cv = [2, 1, -1, math.pi]
    def pv(x, co):
        x = to_quad(x)
        r = zero()
        for c in co:
            r = mul(r, x)
            r = add(r, to_quad(c))
        return to_float(r)

    D  = len(cv) - 1
    N  = 10000
    sc = 1#1e-5
    xv = [x * sc for x in range(N)]
    yv = [pv(x, cv) for x in xv]
    wv = [1. for _ in yv]
    #wv = [y ** -2 for y in yv]
    plan = PolyfitPlan(D, xv, wv)
    fit  = plan.fit(yv)
    print("x=0 ", fit(0))
    print("cof0", fit.coefs(0))
    print("cofs", fit.coefs(xv[N >> 1]))
    print("rms ", fit.rms_errors())
    erel, eres = -1., -1.
    for i, x in enumerate(xv):
        exp  = yv[i]
        obs  = to_float(fit(x, nder=0))
        res  = abs(obs - exp)
        rel  = abs(obs / exp - 1.)
        erel = max(rel, erel)
        eres = max(res, eres)
    print("res ", eres)
    print("rel ", erel)

if __name__ == "__main__":
    demo()
