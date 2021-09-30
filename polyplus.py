"""
integrals and gaussian quadrature add-ons for polyfit.py
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace,useless-object-inheritance

from polyfit import (
    zero, one, to_quad, to_float,
    add, sub, div, mul,
    vappend, vectorsum
)

__all__ = ["PolyplusIntegrator", "PolyplusQuadrature"]

## {{{ integration
class PolyplusIntegrator(object):
    """
    this class computes the definite integral of a fitted
    polynomial. be careful: we have to compute coefficients
    in powers of x to make this work, and those coefficients
    might be less accurate than desired.
    """

    def __init__(self, fit, deg=-1):
        """
        create an integrator from a fit polynomial of given
        degree.
        """
        self._coefs = coefs = fit.evaluator().qcoefs(0., deg)
        deg = len(coefs) - 1
        uno = one()
        for j in range(deg, -1, -1):
            i = deg - j
            fac = div(uno, to_quad(j + 1))
            coefs[i] = mul(coefs[i], fac)
        ## there is an implied-zero constant term

    def qcoefs(self):
        """
        return the coefficients for the integrated polynomial
        in quad precision.
        """
        return self._coefs + [zero()]

    def coefs(self):
        "same as qcoefs, but in double precision"
        return [to_float(c) for c in self.qcoefs()]

    def qintegral(self, x):
        """
        return the quad-precision definite integral from 0 to x.
        """
        x   = to_quad(x)
        ret = zero()
        for c in self._coefs:
            ret = add(mul(ret, x), c)
        ## handle the implied zero constant term
        ret = mul(ret, x)
        return ret

    def __call__(self, x):
        """
        return the double precision definite integral from 0
        to x.
        """
        return to_float(self.qintegral(x))
## }}}
## {{{ quad precision root finding using bisection
def bis(    ## pylint: disable=too-many-arguments
        func, a, fa, b, fb,
        maxiter=108,    ## rel err >= 2**-107
    ):
    """
    quad precision root finding using bisection
    """
    ## this is over the top but works for its use
    ## in this module; don't use this for general
    ## root finding!
    assert to_float(mul(fa, fb)) < 0
    half = to_quad(0.5)
    for _ in range(maxiter):
        c = mul(half, add(a, b))
        if c == a or c == b:
            break
        fc = func(c)
        if fc == fa or fc == fb:
            break
        if to_float(mul(fa, fc)) < 0:
            b, fb = c, fc
        elif fc == (0, 0):
            break
        else:
            a, fa = c, fc
    return c
## }}}
## {{{ quadrature over wv[] and xv[]
class PolyplusQuadrature(object):
    "compute the roots of the orthogonal polynomials"

    def __init__(self, plan):
        "init self from a plan"
        p = plan.ll_plan()
        self.b, self.c, self.g = p["b"], p["c"], p["g"]

        x0, x1 = min(p["x"]), max(p["x"])
        self.x0, self.x1 = to_quad(x0), to_quad(x1)

        self.the_roots   = { 0: [ ] }
        self.the_schemes = { }

    def phi_k(self, x, k):
        "compute the k-th orthogonal poly at x"
        x    = to_quad(x)
        b    = self.b
        c    = self.c
        pjm1 = zero()
        pj   = one()
        for j in range(k):
            pjp1 = sub(
                mul(sub(x, b[j]), pj),
                mul(c[j], pjm1)
            )
            pjm1 = pj
            pj   = pjp1
        return pj

    def __call__(self, func, deg):
        r"""
        compute \sum_{i=1}^N w_i func(x_i) using the quadrature
        scheme \sum_{i=0}^{deg} H_i func(z_i). this is exact if
        func() is a poly of degree < 2D. func is assumed to take
        a float argument.
        """
        f = lambda x: func(to_float(x))
        return to_float(self.qquad(f, deg))

    def qquad(self, func, deg):
        r"""
        compute \sum_{i=1}^N w_i func(x_i) using the quadrature
        scheme \sum_{i=0}^{deg} H_i func(z_i). this is exact if
        func() is a poly of degree < 2D. func is assumed to take
        a quad argument.
        """
        v = [ ]
        for z, H in self.scheme(deg):
            vappend(v, mul(H, to_quad(func(z))))
        return vectorsum(v)

    def roots(self, k):
        "return the roots of the orthogonal poly of degree k"
        if k not in self.the_roots:
            self.the_roots[k] = self._roots(k)
        return self.the_roots[k]

    def scheme(self, k):
        """
        return the ordinates and christoffel numbers for
        the quadrature scheme of order k
        """
        if k not in self.the_schemes:
            self.the_schemes[k] = self._scheme(k)
        return self.the_schemes[k]

    def _roots(self, k):
        "compute the roots using the separation property"
        ranges = [self.x0] + self.roots(k - 1) + [self.x1]
        ret    = [ ]
        func   = lambda x: self.phi_k(x, k)
        for i in range(len(ranges) - 1):
            a  = ranges[i]
            fa = func(a)
            b  = ranges[i + 1]
            fb = func(b)
            ret.append(bis(func, a, fa, b, fb))
        return ret

    def _scheme(self, k):
        """
        compute the quadrature scheme of order k using the
        christoffel-darboux identity
        """
        b   = self.b
        c   = self.c
        g   = self.g
        uno = one()
        ret = [ ]
        for r in self.roots(k):
            v    = [ ]
            pjm1 = zero()
            pj   = one()
            ## loop over polys
            for j in range(k):
                ## add in the next term
                u = div(mul(pj, pj), g[j + 1])
                vappend(v, u)
                ## compute the next poly
                pjp1 = sub(
                    mul(sub(r, b[j]), pj),
                    mul(c[j], pjm1)
                )
                pjm1 = pj
                pj   = pjp1
            ## save the root and christoffel number
            ret.append(
                (r, div(uno, vectorsum(v)))
            )
        return ret
## }}}

## EOF
