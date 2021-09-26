"""
wrapper around libpolyfit.so

there are no comments in this code. see the pure python
code for the commented reference implementation. see the
demo in examples/ex4.py for usage
"""

## {{{ prologue
from __future__ import print_function

## pylint: disable=invalid-name,bad-whitespace,useless-object-inheritance

import array
import ctypes
import os

__all__ = ["PolyfitPlan", "PolyfitFit", "PolyfitEvaluator"]
## }}}
## {{{ util funcs
def isarray(a, minelts=None, maxelts=None):
    "make sure we have a correct array.array"
    if not isinstance(a, array.array):
        raise TypeError("expected array")
    if a.typecode != "d":
        raise TypeError("expected type-d array")
    n = len(a)
    if minelts is not None and n < minelts:
        raise ValueError("array too short")
    if maxelts is not None and n > maxelts:
        raise ValueError("array too long")
    return a

def oserr():
    "raise OSError using errno"
    err = ctypes.get_errno()
    raise OSError(err, os.strerror(err))
## }}}
## {{{ low level glue
_libpolyfit = ctypes.CDLL(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "libpolyfit.so"
    )
)

## {{{ polyfit_free
_polyfit_free = _libpolyfit.polyfit_free
_polyfit_free.argtypes = [ctypes.c_void_p]
_polyfit_free.restype  = None

def polyfit_free(fit):
    "free fit data"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    _polyfit_free(fit)
## }}}
## {{{ polyfit_plan
_polyfit_plan = _libpolyfit.polyfit_plan
_polyfit_plan.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_polyfit_plan.restype  = ctypes.c_void_p

def polyfit_plan(degree, xv, wv):
    "allocate data for a fit"
    isarray(xv, 1)
    N = len(xv)
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    if not 0 <= degree < N:
        raise ValueError("bad degree")
    isarray(wv, N, N)
    if min(wv) <= 0:
        raise ValueError("bad wv")
    xa, _ = xv.buffer_info()
    wa, _ = wv.buffer_info()
    plan  = _polyfit_plan(degree, xa, wa, N)
    if not plan:
        oserr()
    return plan
## }}}
## {{{ polyfit_fit
_polyfit_fit = _libpolyfit.polyfit_fit
_polyfit_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p
]
_polyfit_fit.restype = ctypes.c_void_p

def polyfit_fit(plan, yv):
    "fit a poly to data"
    if not isinstance(plan, int):
        raise TypeError("bad plan type")
    N = polyfit_npoints(plan)
    isarray(yv, N, N)
    ya, _ = yv.buffer_info()
    fit   = _polyfit_fit(plan, ya)
    if not fit:
        oserr()
    return fit
## }}}
## {{{ polyfit_evaluator
_polyfit_evaluator = _libpolyfit.polyfit_evaluator
_polyfit_evaluator.argtypes = [ctypes.c_void_p]
_polyfit_evaluator.restype  = ctypes.c_void_p

def polyfit_evaluator(fit):
    "return an evaluator"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    ret = _polyfit_evaluator(fit)
    if not ret:
        oserr()
    return ret
## }}}
## {{{ polyfit_eval
_polyfit_eval = _libpolyfit.polyfit_eval
_polyfit_eval.argtypes = [
    ctypes.c_void_p, ctypes.c_double, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int
]
_polyfit_eval.restype = ctypes.c_int

def polyfit_eval(evaluator, x, degree, nderiv):
    "eval poly and derivatives at x"
    if not isinstance(evaluator, int):
        raise TypeError("expected an evaluator")
    if not isinstance(x, float):
        raise TypeError("x must be a float")
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    if not isinstance(nderiv, int):
        raise TypeError("degree must be an int")
    D = polyfit_maxdeg(evaluator)
    if degree > D:
        raise ValueError("bad degree")
    if degree < 0:
        degree = D
    if nderiv < 0:
        nderiv = degree
    ret   = array.array("d", [0] * (1 + nderiv))
    ra, _ = ret.buffer_info()
    if _polyfit_eval(evaluator, x, degree, ra, nderiv) < 0:
        oserr()
    return ret
## }}}
## {{{ polyfit_coefs
_polyfit_coefs = _libpolyfit.polyfit_coefs
_polyfit_coefs.argtypes = [
    ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_void_p
]
_polyfit_coefs.restype = ctypes.c_int

def polyfit_coefs(evaluator, x, degree):
    "coefs poly and derivatives at x"
    if not isinstance(evaluator, int):
        raise TypeError("expected an evaluator")
    if not isinstance(x, float):
        raise TypeError("x must be a float")
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    D = polyfit_maxdeg(evaluator)
    if degree > D:
        raise ValueError("bad degree")
    if degree < 0:
        degree = D
    ret   = array.array("d", [0] * (1 + degree))
    ra, _ = ret.buffer_info()
    _polyfit_coefs(evaluator, x, degree, ra)
    return ret
## }}}
## {{{ polyfit_npoints
_polyfit_npoints = _libpolyfit.polyfit_npoints
_polyfit_npoints.argtypes = [ctypes.c_void_p]
_polyfit_npoints.restype  = ctypes.c_int

def polyfit_npoints(obj):
    "return #points in fit"
    if not isinstance(obj, int):
        raise TypeError("bad fit type")
    return _polyfit_npoints(obj)
## }}}
## {{{ polyfit_maxdeg
_polyfit_maxdeg = _libpolyfit.polyfit_maxdeg
_polyfit_maxdeg.argtypes = [ctypes.c_void_p]
_polyfit_maxdeg.restype  = ctypes.c_int

def polyfit_maxdeg(obj):
    "return max degree of fit"
    if not isinstance(obj, int):
        raise TypeError("bad fit type")
    return _polyfit_maxdeg(obj)
## }}}
## {{{ polyfit_resids
_polyfit_res = _libpolyfit.polyfit_resids
_polyfit_res.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_polyfit_res.restype  = ctypes.c_void_p

def polyfit_resids(fit):
    "return residuals for the maxdeg fit"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    ret   = array.array("d", [0] * polyfit_npoints(fit))
    ra, _ = ret.buffer_info()
    _polyfit_res(fit, ra)
    return ret
## }}}
## {{{ polyfit_rms_errs
_polyfit_rms_errs = _libpolyfit.polyfit_rms_errs
_polyfit_rms_errs.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_polyfit_rms_errs.restype  = ctypes.c_void_p

def polyfit_rms_errs(fit):
    "return rms fit errors per degree"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    ret   = array.array("d", [0] * (polyfit_maxdeg(fit) + 1))
    ra, _ = ret.buffer_info()
    _polyfit_rms_errs(fit, ra)
    return ret
## }}}
## }}}
## {{{ class-based interface
class PolyfitEvaluator(object):
    """
    returned by PolyfitFit.evaluator(). this object evaluates
    the fit polynomial and its derivatives, and also returns
    its coefficients in powers of (x - x0) for given x0.
    """

    def __init__(self, fit):
        self.eval = polyfit_evaluator(fit)

    def __del__(self):
        "deallocate on destruct"
        p, self.eval = self.eval, None
        if p:
            try:
                polyfit_free(p)
            except NameError:
                pass

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
        ret = polyfit_eval(self.eval, x, deg, nder)
        return ret[0] if len(ret) == 1 else ret

    def coefs(self, x0, deg=-1):
        """
        return the coefficients of the fit polynomial of degree
        deg about (x - x0). if deg is negative, use maxdeg
        instead.
        """
        return polyfit_coefs(self.eval, x0, deg)

class PolyfitFit(object):
    """
    orthogonal polynomial fitter returned by PolyfitPlan.fit()
    """

    def __init__(self, plan, yv):
        self.fit = polyfit_fit(plan, yv)

    def __del__(self):
        "deallocate on destruct"
        p, self.fit = self.fit, None
        if p:
            try:
                polyfit_free(p)
            except NameError:
                pass

    def evaluator(self):
        """
        return a PolyfitEvaluator for this fit.
        """
        return PolyfitEvaluator(self.fit)

    def residuals(self):
        """
        return the list of residuals for the maxdeg fit.
        """
        return polyfit_resids(self.fit)

    def rms_errors(self):
        """
        return a list of rms errors, one per fit degree. use them
        to detect overfitting.
        """
        return polyfit_rms_errs(self.fit)

class PolyfitPlan(object):
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

    def __del__(self):
        "deallocate on destruct"
        p, self.plan = self.plan, None
        if p:
            try:
                polyfit_free(p)
            except NameError:
                pass

    def fit(self, yv):
        """
        given a set of y values in yv[], compute all least
        squares fits to yv[] up to degree maxdeg. returns
        a PolyfitFit object.
        """
        return PolyfitFit(self.plan, yv)

    def maxdeg(self):
        "return the maximum fit degree"
        return polyfit_maxdeg(self.plan)

    def npoints(self):
        "return the number of fit points"
        return polyfit_npoints(self.plan)
## }}}

## EOF
