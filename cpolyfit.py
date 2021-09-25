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
import time

__all__ = ["Polyfit"]
## }}}
## {{{ util funcs
def isarray(a, minelts=None, maxelts=None):
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

_free = _libpolyfit.polyfit_free
_free.argtypes = [ctypes.c_void_p]
_free.restype  = None

def polyfit_free(fit):
    "free fit data"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    _free(fit)

_plan = _libpolyfit.polyfit_plan
_plan.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_plan.restype  = ctypes.c_void_p

def polyfit_plan(degree, xv, wv):
    "allocate data for a fit"
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    if not 0 <= degree < npoints:
        raise ValueError("bad values")
    isarray(xv, 1)
    N = len(xv)
    isarray(wv, N, N)
    if min(wv) <= 0:
        raise ValueError("bad wv")
    xa, _ = xv.buffer_info()
    wa, _ = wv.buffer_info()
    return _plan(degree, xa, wa, npoints)

_fit = _libpolyfit.polyfit_fit
_fit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p
]
_fit.restype = ctypes.c_void_p

def polyfit_fit(plan, yv):
    "fit a poly to data"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    N = polyfit_npoints(plan)
    isarray(yv, N, N)
    ya, _ = yv.buffer_info()
    return _polyfit_fit(plan, ya)

_eval = _libpolyfit.polyfit_eval
_eval.argtypes = [ctypes.c_void_p]
_eval.restype  = ctypes.c_void_p

def polyfit_evaluator(fit):
    "eval fit data"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    return _eval(fit)

_poly_npoints = _libpolyfit.polyfit_npoints
_poly_npoints.argtypes = [ctypes.c_void_p]
_poly_npoints.restype  = ctypes.c_int

def polyfit_npoints(fit):
    "return #points in fit"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    return _poly_npoints(fit)

_poly_maxdeg = _libpolyfit.polyfit_maxdeg
_poly_maxdeg.argtypes = [ctypes.c_void_p]
_poly_maxdeg.restype  = ctypes.c_int

def polyfit_maxdeg(fit):
    "return max degree of fit"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    return _poly_maxdeg(fit)

_polyres = _libpolyfit.polyfit_resids
_polyres.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_polyres.restype  = ctypes.c_void_p

def polyfit_resids(fit):
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    ret   = array.array("d", [0] * polyfit_npoints(fit))
    ra, _ = ret.buffer_info()
    _polyres(fit, ra)
    return ret

_polyerr = _libpolyfit.polyfit_rms_errs
_polyerr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_polyerr.restype  = ctypes.c_void_p

def polyfit_rms_errs(fit):
    "return rms fit errors per degree"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    ret   = array.array("d", [0] * (polyfit_maxdeg(fit) + 1))
    ra, _ = ret.buffer_info()
    _polyerr(fit, ra)
    return ret

## }}}
## {{{ class-based interface
class Polyfit(object):
    "polynomial fitting class"

    def __init__(self, maxdeg, xv, yv, wv):
        if not isinstance(xv, array.array):
            xv = array.array('d', xv)
        if not isinstance(yv, array.array):
            yv = array.array('d', yv)
        if not isinstance(wv, array.array):
            wv = array.array('d', wv)
        self.xv    = xv
        self.yv    = yv
        self.wv    = wv
        self._fit  = polyfit_alloc(len(xv), maxdeg)
        self._time = polyfit(self._fit, xv, yv, wv)

    def __call__(self, x, degree=None, nderiv=0):
        "evaluate poly and its derivatives"
        if degree is None:
            degree = self.maxdeg()
        nd = self.maxdeg() if nderiv < 0 else nderiv
        ret = array.array('d', [0] * (nd + 1))
        polyfit_val(self._fit, x, degree, ret)
        return ret if nderiv else ret[0]

    def close(self):
        "finalize"
        f, self._fit = self._fit, None
        if f is not None:
            try:
                poly_free(f)
            except NameError:
                pass

    __del__ = close

    def coefs(self, degree=None, x0=0):
        "return coefs about x0"
        if degree is None:
            degree = self.maxdeg()
        ret = array.array('d', [0] * (degree + 1))
        polyfit_cofs(self._fit, degree, x0, ret)
        return ret

    def maxdeg(self):
        "return max fit degree"
        return polyfit_maxdeg(self._fit)

    def npoints(self):
        "return number of points the fit"
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
        "return rms error for given degree"
        if degree is None:
            degree = self.maxdeg()
        return polyfit_err(self._fit, degree)

    def runtime(self):
        "return the time it took to perform the fit"
        return self._time
## }}}

## EOF
