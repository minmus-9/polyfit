""" integrals and gaussian quadrature add-ons for polyfit.py
"""

from __future__ import print_function as _

## pylint: disable=invalid-name,bad-whitespace

from polyfit import (
    zero,
    one,
    to_dd,
    to_float,
    add,
    sub,
    div,
    mul,
    vappend,
    vectorsum,
    PolyfitBase,
)

__all__ = ["PolyplusIntegrator", "PolyplusQuadrature"]

## {{{ integration
class PolyplusIntegrator(PolyfitBase):
    """
    this class computes the definite integral of a fitted
    polynomial. be careful: we have to compute coefficients
    in powers of x to make this work, and those coefficients
    might be less accurate than desired.
    """

    def __init__(self, data, deg=-1):
        """
        create an integrator from a fit polynomial of given
        degree.
        """
        if isinstance(data, dict):
            self.data = data
            self._coefs = data["coefs"]
        else:
            self.data = data.to_data()["fit"].copy()
            self._coefs = coefs = data.evaluator().coefs(zero(), deg)
            self.data["coefs"] = coefs

            deg = len(coefs) - 1
            uno = one()
            for j in range(deg, -1, -1):
                i = deg - j
                fac = div(uno, to_dd(j + 1))
                coefs[i] = mul(coefs[i], fac)
            ## there is an implied-zero constant term

    def qcoefs(self):
        """
        return the coefficients for the integrated polynomial
        in double-double precision.
        """
        return self._coefs + [zero()]

    def coefs(self):
        "same as qcoefs, but in double precision"
        return [to_float(c) for c in self.qcoefs()]

    def __call__(self, x):
        """
        return the double-double precision definite integral
        from 0 to x.  the return value is double-double if x
        is double-double.
        """
        q = to_dd(x)
        ret = zero()
        for c in self._coefs:
            ret = add(mul(ret, q), c)
        ## handle the implied zero constant term
        ret = mul(ret, q)
        return ret if isinstance(x, tuple) else to_float(ret)


## }}}
## {{{ double-double precision root finding using bisection
def bis(  ## pylint: disable=too-many-arguments
    func,
    a,
    fa,
    b,
    fb,
    maxiter=108,  ## rel err >= 2**-107
):
    """
    double-double precision root finding using bisection
    """
    ## this is over the top but works for its use
    ## in this module; don't use this for general
    ## root finding!
    assert to_float(mul(fa, fb)) < 0
    half = to_dd(0.5)
    for _ in range(maxiter):
        c = mul(half, add(a, b))
        if c in (a, b):
            break
        fc = func(c)
        if fc in (fa, fb):
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
class PolyplusQuadrature(PolyfitBase):
    ## pylint: disable=too-many-instance-attributes
    """
    this class implements gaussian quadrature on a
    discrete set of points.

    if you want to compute

        sum(f(x_i) * w_i for x_i, w_i in zip(xv, wv))

    you can replace this with gaussian quadrature as

        sum(f(z_i) * H_i for z_i, H_i in zip(Z, H))

    the difference is that D = len(Z) is much smaller
    than len(xv). Z and H are generated from a fit
    plan for xv and wv. this module provides a function
    to accurately sum f().

    the quadrature formula is exact for polynomial
    functions f() up to degree 2D-1.

    for non-polynomial functions, the error is
    proportional to the 2D-th derivative of f()
    divided by (2D)!
    """

    def __init__(self, data):
        "init self from a plan"
        if isinstance(data, dict):
            self.data = data
        else:
            self.data = data = data.to_data()["plan"].copy()

            data["x0"], data["x1"] = min(data["x"]), max(data["x"])

            data["roots"] = {0: []}
            data["schemes"] = {}

        self.b, self.c, self.g = data["b"], data["c"], data["g"]
        self.x0, self.x1 = to_dd(data["x0"]), to_dd(data["x1"])
        self.the_roots = data["roots"]
        self.the_schemes = data["schemes"]

    def __call__(self, func, deg):
        r"""
        compute \sum_{i=1}^N w_i func(x_i) using the quadrature
        scheme \sum_{i=0}^{deg} H_i func(z_i). this is exact if
        func() is a poly of degree < 2D. func is assumed to take
        and return a float.
        """
        f = lambda x: func(to_float(x))
        return to_float(self.qquad(f, deg))

    def qquad(self, func, deg):
        r"""
        compute \sum_{i=1}^N w_i func(x_i) using the quadrature
        scheme \sum_{i=0}^{deg} H_i func(z_i). this is exact if
        func() is a poly of degree < 2D. func is assumed to take
        and return a double-double.
        """
        v = []
        for z, H in self.scheme(deg):
            vappend(v, mul(H, to_dd(func(z))))
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

    def _phi_k(self, x, k):
        "internal: compute the k-th orthogonal poly at x"
        x = to_dd(x)
        b = self.b
        c = self.c
        pjm1 = zero()
        pj = one()
        for j in range(k):
            pjp1 = sub(mul(sub(x, b[j]), pj), mul(c[j], pjm1))
            pjm1 = pj
            pj = pjp1
        return pj

    def _roots(self, k):
        r"""
        internal:

        compute the roots of \phi_k using the separation
        property
        """
        ranges = [self.x0] + self.roots(k - 1) + [self.x1]
        ret = []
        func = lambda x: self._phi_k(x, k)
        for i in range(len(ranges) - 1):
            a = ranges[i]
            fa = func(a)
            b = ranges[i + 1]
            fb = func(b)
            ret.append(bis(func, a, fa, b, fb))
        return ret

    def _scheme(self, k):
        """
        internal:

        compute the quadrature scheme of order k using the
        christoffel-darboux identity
        """
        b = self.b
        c = self.c
        g = self.g
        uno = one()
        ret = []
        for r in self.roots(k):
            v = []
            pjm1 = zero()
            pj = one()
            ## loop over polys
            for j in range(k):
                ## add in the next term
                u = div(mul(pj, pj), g[j + 1])
                vappend(v, u)
                ## compute the next poly
                pjp1 = sub(mul(sub(r, b[j]), pj), mul(c[j], pjm1))
                pjm1 = pj
                pj = pjp1
            ## save the root and christoffel number
            ret.append((r, div(uno, vectorsum(v))))
        return ret


## }}}

## EOF
