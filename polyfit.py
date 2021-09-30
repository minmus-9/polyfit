#!/usr/bin/env python3

"quad precision orthogonal polynomial least squares fitting"

## {{{ prologue
from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace
## pylint: disable=useless-object-inheritance
## pylint: disable=unnecessary-comprehension

import math

__all__ = [
    "PolyfitPlan", "PolyfitFit", "PolyfitEvaluator",
    "polyfit_plan", "polyfit_fit", "polyfit_eval",
    "polyfit_coefs", "polyfit_maxdeg", "polyfit_npoints",
    "polyfit_qeval", "polyfit_qcoefs",

    "zero", "one", "vappend", "vectorsum", "to_quad",
    "to_float", "add", "sub", "mul", "div", "sqrt",
]
## }}}
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
    return sum2s(sumkcore(p, K))

def vectorsum(vec):
    "accurately sum a vector of floats, 19n-13 flops"
    return sumk(vec, K=3)
## }}}
## {{{ utility functions
def zero():
    "return quad precision 0"
    return (0., 0.)

def one():
    "return quad precision 1"
    return (1., 0.)

def vappend(vec, x):
    """
    append quad precision number to vector. this is used
    for vectorsum():

        v = [ ]
        for x in y:
            quad = ...
            vappend(v, quad)
        s = vectorsum(v)

    vectorsum() is more accurate than using add() in a loop.
    """
    vec.extend(x)

def to_quad(x):
    "convert float or quad to quad"
    return x if isinstance(x, tuple) else (float(x), 0.)

def to_float(x):
    "convert quad to float"
    return x[0] if isinstance(x, tuple) else float(x)
## }}}
## {{{ quad precision arithmetic
def add(x, y):
    "add two quads, 14 flops"
    x, xx = x
    y, yy = y
    z, zz = twosum(x, y)
    return twosum(z, zz + xx + yy)

def sub(x, y):
    "subtract 2 quads, 14 flops"
    x, xx = x
    y, yy = y
    z, zz = twodiff(x, y)
    return twosum(z, zz + xx - yy)

def mul(x, y):
    "multiply 2 quads, 33 flops"
    x, xx = x
    y, yy = y
    z, zz = twoproduct(x, y)
    zz   += xx * y + x * yy
    return twosum(z, zz)

def div(x, y):
    "divide 2 quads, 36 flops, from dekker"
    x, xx = x
    y, yy = y
    c     = x / y
    u, uu = twoproduct(c, y)
    cc    = (x - u - uu + xx - c * yy) / y
    return twosum(c, cc)

def sqrt(x):
    "square root of a quad, 35 flops, from dekker"
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

    returns a plan object than can be json-serialized.
    """
    ## pylint: disable=too-many-locals

    ## convert to quad
    xv = [to_quad(x) for x in xv]
    wv = [to_quad(w) for w in wv]
    ## build workspaces and result object
    N = len(xv)
    b = [ ]             ## recurrence coefs b_k
    c = [ ]             ## recurrence coefs c_k
    g = [one()]         ## \gamma_k^2 \equiv (\phi_k, \phi_k)
    r = {
        "D": maxdeg,    ## max fit degree
        "N": N,         ## number of data points
        "b": b,         ## coefficients b_k
        "c": c,         ## coefficients c_k
        "g": g,         ## normalization constants g_k
        "x": xv,        ## x values, needed for polyfit_fit
        "w": wv         ## y values, needed for polyfit_fit
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
        ## if we aren't done, update pk[] and pkm1[]
        ## for the next round
        if k != maxdeg:
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

    returns a json-serializable fit data object.
    """
    ## pylint: disable=too-many-locals
    N, D = plan["N"], plan["D"]
    b, c = plan["b"], plan["c"]
    g    = plan["g"]
    wv   = plan["w"]
    xv   = plan["x"]

    a, e = [ ], [ ]                 ## fit coefs and rms errors
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


        ## if we aren't done, update pk[] and pkm1[]
        ## for the next round
        if k != D:
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
    ## return fit data
    return {
        "a": a,     ## orthogonal poly coefs
        "e": e,     ## per-degree rms errors
        "r": rv     ## per-point residuals
    }
## }}}
## {{{ polyfit_eval
def polyfit_qeval(plan, ll_fit, x, deg=-1, nder=0):
    """
    given a plan, a fit data object returned by
    polyfit_fit, a point x, a least squares fit degree deg,
    and a desired number of derivatives to compute nder,
    calculate and return the value of the polynomial and
    any requested derivatives.

    if deg is negative, use maxdeg instead. if nder is
    negative, use the final value of deg; otherwise, compute
    ndeg derivatives of the least squares polynomial of
    degree deg.

    returns a list of quads whose first element is the value
    of the least squares polynomial of degree deg at x.
    subsequent elements are the requested derivatives. if zero
    derivatives are requested, the scalar quad function value
    is returned.
    """
    ## pylint: disable=too-many-locals
    a, b, c, D = ll_fit["a"], plan["b"], plan["c"], plan["D"]

    if deg < 0:
        deg = D
    if nder < 0:
        nder = deg

    ## z_k^{(j-1)} and z_k^{(j)} for clenshaw's recurrence
    zjm1 = a[:deg+1] + [zero(), zero()] ## init to a_kj
    zj   = [zero()] * (deg + 3)

    fac  = one()            ## j! factor
    zeds = [zero(), zero()]
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
            tmp = sub(
                mul(sub(x, b[t]), zj[k + 1]),
                mul(c[t + 1], zj[k + 2])
            )
            zj[k] = add(zjm1[k], tmp)
        ## save j! z_j^{(j)}
        ret.append(mul(fac, zj[j]))
        ## update z if we aren't done
        if j != lim:
            ## update zjm1
            zjm1[:] = zj
            ## zj only needs last 2 elements cleared
            zj[-2:] = zeds
    if nder > deg:
        ret += [zero()] * (nder - deg)
    ## returns quad precision (for polyfit_coefs)
    return ret

def polyfit_eval(plan, fit, x, deg=-1, nder=0):
    """
    given a plan, a fit data object returned by
    polyfit_fit, a point x, a least squares fit degree deg,
    and a desired number of derivatives to compute nder,
    calculate and return the value of the polynomial and
    any requested derivatives.

    if deg is negative, use maxdeg instead. if nder is
    negative, use the final value of deg; otherwise, compute
    ndeg derivatives of the least squares polynomial of
    degree deg.

    returns a list whose first element is the value of the
    least squares polynomial of degree deg at x. subsequent
    elements are the requested derivatives. if zero
    derivatives are requested, the scalar function value is
    returned.
    """
    ## get float values
    r = polyfit_qeval(plan, fit, x, deg, nder)
    r = [to_float(v) for v in r]
    ## return scalar if no derivs
    return r[0] if len(r) == 1 else r
## }}}
## {{{ polyfit_coefs
def polyfit_qcoefs(plan, ll_fit, x0=0., deg=-1):
    """
    given a plan, a set of expansion coefficients generated
    by polyfit_fit, a center point x0, and a least squares
    fit degree, return the coefficients of powers of (x - x0)
    with the highest powers first. if deg is negative (the
    default), use maxdeg instead. the coefficients are quad
    precision.
    """
    ## get value and derivs, divide by j!
    vals = polyfit_qeval(plan, ll_fit, x0, deg, deg)
    fac  = one()
    for j in range(2, len(vals)):
        fac     = div(fac, to_quad(j))
        vals[j] = mul(vals[j], fac)
    ## get highest power first
    vals.reverse()
    return vals

def polyfit_coefs(plan, ll_fit, x0=0., deg=-1):
    """
    given a plan, a set of expansion coefficients generated
    by polyfit_fit, a center point x0, and a least squares
    fit degree, return the coefficients of powers of (x - x0)
    with the highest powers first. if deg is negative (the
    default), use maxdeg instead.
    """
    vals = polyfit_qcoefs(plan, ll_fit, x0, deg)
    return [to_float(v) for v in vals]
## }}}
## {{{ polyfit_maxdeg
def polyfit_maxdeg(plan):
    "return the maximum possible fit degree"
    return plan["D"]
## }}}
## {{{ polyfit_npoints
def polyfit_npoints(plan):
    "return the number of data points being fit"
    return plan["N"]
## }}}
## {{{ Polyfit classes
class PolyfitBase(object):
    "base class for polyfit classes"
    ## pylint: disable=too-few-public-methods

    def close(self):
        "deallocate resources, a no-op for this impementation"

class PolyfitEvaluator(PolyfitBase):
    """
    returned by PolyfitFit.evaluator(). this object evaluates
    the fit polynomial and its derivatives, and also returns
    its coefficients in powers of (x - x0) for given x0.
    """

    def __init__(self, plan, fit):
        self.plan, self.fit = plan, fit

    def __call__(self, x, deg=-1, nder=0):
        """
        given a point x, a least squares fit degree deg,
        and a desired number of derivatives to compute nder,
        calculate and return the value of the polynomial and
        any requested derivatives.

        if deg is negative, use maxdeg instead. if nder is
        negative, use the final value of deg; otherwise, compute
        nder derivatives of the least squares polynomial of
        degree deg.

        returns a list whose first element is the value of the
        least squares polynomial of degree deg at x. subsequent
        elements are the requested derivatives. if zero
        derivatives are requested, the scalar function value is
        returned.
        """
        return polyfit_eval(self.plan, self.fit, x, deg, nder)

    def coefs(self, x0, deg=-1):
        """
        return the coefficients of the fit polynomial of degree
        deg about (x - x0). if deg is negative, use maxdeg
        instead.
        """
        return polyfit_coefs(self.plan, self.fit, x0, deg)

    def qcoefs(self, x0, deg=-1):
        "like coefs() but return quad-precision ones"
        return polyfit_qcoefs(self.plan, self.fit, x0, deg)

class PolyfitFit(PolyfitBase):
    """
    orthogonal polynomial fitter returned by PolyfitPlan.fit()
    """

    def __init__(self, plan, yv):
        self.plan = plan
        self.fit  = fit = polyfit_fit(plan, yv)
        self.res  = [to_float(r) for r in fit["r"]]
        self.rms  = [to_float(e) for e in fit["e"]]

    def evaluator(self):
        """
        return a PolyfitEvaluator for this fit.
        """
        return PolyfitEvaluator(self.plan, self.fit)

    def residuals(self):
        """
        return the list of residuals for the maxdeg fit.
        """
        return self.res

    def rms_errors(self):
        """
        return a list of rms errors, one per fit degree. use them
        to detect overfitting.
        """
        return self.rms

class PolyfitPlan(PolyfitBase):
    """
    orthogonal polynomial least squares planning class. you must
    create one of these prior to fitting; it can be reused for
    multiple fits of the same xv[] and wv[].
    """

    def __init__(self, maxdeg, xv, wv):
        """
        given x values in xv[] and positive weights in wv[],
        make a plan to perform least squares fitting up to
        degree maxdeg.

        this is code for "compute everything need to calculate
        an expansion in xv- and wv-specific orthogonal
        polynomials".
        """
        self.plan = polyfit_plan(maxdeg, xv, wv)

    def fit(self, yv):
        """
        given a set of y values in yv[], compute all least
        squares fits to yv[] up to degree maxdeg. returns
        a PolyfitFit object.
        """
        return PolyfitFit(self.plan, yv)

    def ll_plan(self):
        """
        return the low-level plan created by polyfit_plan()
        """
        return self.plan

    def maxdeg(self):
        "return the maximum fit degree"
        return polyfit_maxdeg(self.plan)

    def npoints(self):
        "return the number of fit points"
        return polyfit_npoints(self.plan)
## }}}

## EOF
