"""
quad precision orthogonal polynomial least squares fit

see polyfit.pdf and the code in examples/
"""

## {{{ prologue
from __future__ import print_function

## pylint: disable=invalid-name,bad-whitespace,useless-object-inheritance

import math
import time

__all__ = ["Polyfit"]
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
    p = sumkcore(p, K)
    return sum2s(p)

def vectorsum(vec):
    "19n-13 flops, sumk() with K=3"
    return sumk(vec, K=3)
## }}}
## {{{ utility functions
def vappend(vec, x):
    "append quad to vector"
    vec.extend(x)

def zero():
    "yup"
    return (0., 0.)

def one():
    "yup"
    return (1., 0.)

def to_quad(x):
    "float to quad"
    return x if isinstance(x, tuple) else (float(x), 0.)

def quad_to_float(x):
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
## {{{ orthogonal polynomial least squares fitting
def polyfit(xv, yv, wv, D):
    ## pylint: disable=too-many-locals
    """
    orthogonal polynomial fit

    given x values xv[], y values yv[], positive weights wv[],
    and a maximum fit degree D, compute the least squares
    fits up to degree D
    """
    ## fit: y_k \approx \sum_{k=0}^D a_k \phi_k(x)

    ## inner product: (f, g) = \sum_{i=0}^{N-1} w_i f(x_i) g(x_i)

    ## recurrence:
    ##      \phi_{k+1}(x) = (x - b_k) \phi_k(x) - c_k \phi_{k-1}(x)

    ## g_k = (\phi_k, \phi_k)
    ## b_k = (x \phi_k, \phi_k) / g_k
    ## c_k = g_k / g_{k-1}
    ## a_k = (y, \phi_k) / g_k
    assert len(xv) == len(yv) == len(wv)
    assert min(wv) > 0
    xv = [to_quad(x) for x in xv]
    rv = [to_quad(y) for y in yv]
    wv = [to_quad(w) for w in wv]
    N  = len(xv)
    a  = [ ]        ## a_k fit coefficients
    b  = [ ]        ## b_k in recurrence
    g  = [one()]    ## g_k poly 2-norm
    c  = [ ]        ## c_k in recurrence
    e  = [ ]        ## rms fit errors

    ret = {         ## fit object
        "a": a,
        "b": b,
        "c": c,
        "e": e,
        "d": D,
        "n": N
    }

    phi_km1 = [zero()] * N  ## \phi_{-1}
    phi_k   = [one()]  * N  ## \phi_0
    for k in range(D+1):  ## pylint: disable=unused-variable
        ## vectors to hold pieces of inner products
        avec = [ ]
        bvec = [ ]
        gvec = [ ]
        ## compute inner products for a_k, b_k, c_k, g_k
        for i in range(N):
            s = mul(wv[i], phi_k[i])
            t = mul(s, phi_k[i])
            ## a_k += wv[i] * rv[i] * phi_k[i]
            vappend(avec, mul(s, rv[i]))
            ## b_k += wv[i] * xv[i] * phi_k[i] * phi_k[i]
            vappend(bvec, mul(t, xv[i]))
            ## g_k += wv[i] * phi_k[i] * phi_k[i]
            vappend(gvec, t)
        ## turn vectors back to scalars and normalize
        g_k = vectorsum(gvec)
        a_k = div(vectorsum(avec), g_k)
        b_k = div(vectorsum(bvec), g_k)
        c_k = div(g_k, g[-1])
        a.append(a_k)
        b.append(b_k)
        c.append(c_k)
        g.append(g_k)

        ## subtract projection a_k \phi_k from yv, leaving the
        ## residuals in rv. dpolft does this and it does actually
        ## help. plus it enables the rms calculation below.
        for i in range(N):
            ## rv[i] -= a_k * phi_k[i]
            rv[i] = sub(rv[i], mul(a_k, phi_k[i]))

        ## compute the (unweighted) rms error in the fit
        evec = [ ]
        for i, r in enumerate(rv):
            ## err += res[i] * res[i]
            vappend(evec, mul(r, r))
        erms = quad_to_float(
            sqrt(div(vectorsum(evec), to_quad(N)))
        )
        e.append(erms)

        ## update polys using recurrence
        if k != D:
            for i in range(N):
                ## \phi_{k+1} = (x - b_k) \phi_k - c_k \phi_{k-1}
                phi_kp1    = sub(
                    mul(sub(xv[i], b_k), phi_k[i]),
                    mul(c_k, phi_km1[i])
                )
                phi_km1[i] = phi_k[i]
                phi_k[i]   = phi_kp1

    c.append(zero())    ## for polyfit_val()

    return ret
## }}}
## {{{ least squares polynomial evaluation
def polyfit_val(fit, x, deg=-1, nderiv=0, extended=False):
    ## pylint: disable=too-many-locals
    """
    return the value of the fit for degree deg and nderiv
    derivatives at the point x. if deg is negative, use the
    highest degree of the fit. if nderiv is negative, compute
    all derivatives of the fit

    returns a list of the function values and its derivatives
    """
    x = to_quad(x)
    a, b, c = fit["a"], fit["b"], fit["c"]
    if deg < 0:
        deg = len(a) - 1
    if nderiv < 0:
        nderiv = deg

    ret  = [ ]
    ## init z^{(j-1)} and z^{(j)}
    zjm1 = a[:deg+2] + [zero(), zero()]
    zj   = [zero()]  * (deg + 3)
    fac  = one()
    for j in range(min(deg, nderiv) + 1):
        if j > 1:
            fac = mul(fac, to_quad(j))
        ## compute the next lowest z_k^{(j)}
        for k in range(deg, j - 1, -1):
            t = k - j
            ## (x-b[t]) * zj[k+1] - c[t+1] * zj[k+2]j
            tmp = sub(
                mul(sub(x, b[t]), zj[k+1]),
                mul(c[t+1], zj[k+2])
            )
            zj[k] = add(zjm1[k], tmp)
        ## save off the function value or derivative
        val = mul(fac, zj[j])
        ret.append(val if extended else quad_to_float(val))
        ## update z vectors
        zjm1 = zj
        zj   = [zero()] * (deg + 3)
    if nderiv > deg:
        ret.extend([zero() if extended else 0.] * (nderiv - deg))
    return ret
## }}}
## {{{ least squares polynomial coefficients
def polyfit_cofs(fit, deg=-1, x0=0., extended=False):
    """
    return taylor coefficients of fit for the given degree deg
    about x0.
    """
    derivs = polyfit_val(fit, x0, deg=deg, nderiv=-1, extended=True)
    fac    = one()
    for i in range(1, len(derivs)):
        fac = div(fac, to_quad(i))
        derivs[i] = mul(derivs[i], fac)
    derivs.reverse()
    return \
        derivs if extended else [quad_to_float(d) for d in derivs]
## }}}
## {{{ least squares polynomial per-degree errors
def polyfit_err(fit, degree):
    "return the rms errors for each fit degree"
    return fit["e"][degree]
## }}}
## {{{ fetch fit params
def polyfit_npoints(fit):
    "return number of data points in fit"
    return fit["n"]

def polyfit_maxdeg(fit):
    "return max degree of fit"
    return fit["d"]
## }}}
## {{{ class-based interface
class Polyfit(object):
    "polynomial fitting class"

    def __init__(self, maxdeg, xv, yv, wv):
        """
        given x- and y-values in xv[] and yv[], along with
        positive fit weights in wv[], compute all least-squares
        fits up to degree maxdeg
        """
        t0         = time.time()
        self.xv    = xv
        self.yv    = yv
        self.wv    = wv
        self._fit  = polyfit(xv, yv, wv, D=maxdeg)
        self._time = time.time() - t0

    def __call__(self, x, degree=None, nderiv=0):
        """
        evaluate poly and (optionally) some of its derivatives.
        if degree is None, return the values for the maximum
        fit degree

        if nderiv is nonzero, return the polynomial value and
        nderiv derivatives as a list; if nderiv is negative,
        return all derivatives up to degree.. if nderiv is 0,
        return the scalar polynomial value. this is the default
        """
        if degree is None:
            degree = self.maxdeg()
        nd  = self.maxdeg() if nderiv < 0 else nderiv
        ret = polyfit_val(self._fit, x, degree, nd)
        return ret if nderiv else ret[0]

    def close(self):
        "finalize self (no-op for this version)"

    def coefs(self, degree=None, x0=0):
        """
        return the coefficients for fit degree degree about
        (x-x0). if degree is None, use the maximum fit degree
        """
        if degree is None:
            degree = self.maxdeg()
        return polyfit_cofs(self._fit, degree, x0)

    def maxdeg(self):
        "return max fit degree"
        return polyfit_maxdeg(self._fit)

    def npoints(self):
        "return number of data points used to perform the fit"
        return polyfit_npoints(self._fit)

    def rel_err(self, degree=None):
        "return the max relative fit error across all x values"
        if degree is None:
            degree = self.maxdeg()
        err = -1.
        for x, exp in zip(self.xv, self.yv):
            obs = self(x, degree=degree)
            if exp:
                rel = abs(obs / exp - 1.)
                if rel > err:
                    err = rel
        return err

    def rms_err(self, degree=None):
        """
        return the residual rms fit residual for degree degree.
        if degree is None, use the maximum fit degree
        """
        if degree is None:
            degree = self.maxdeg()
        return polyfit_err(self._fit, degree)

    def runtime(self):
        "return the time it took to perform the fit"
        return self._time
## }}}

## vim: ft=python
