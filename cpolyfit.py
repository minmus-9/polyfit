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
## {{{ low level glue
_libpolyfit = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "libpolyfit.so")
)

_alloc = _libpolyfit.polyfit_alloc
_alloc.argtypes = [ctypes.c_int, ctypes.c_int]
_alloc.restype  = ctypes.c_void_p

def polyfit_alloc(npoints, degree):
    "allocate data for a fit"
    if not (
            isinstance(npoints, int) and \
            isinstance(degree, int)
        ):
        raise TypeError("bad types")
    if not 0 <= degree < npoints:
        raise ValueError("bad values")
    return _alloc(npoints, degree)

_free = _libpolyfit.polyfit_free
_free.argtypes = [ctypes.c_void_p]
_free.restype  = None

def polyfit_free(fit):
    "free fit data"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    _free(fit)

_polyfit = _libpolyfit.polyfit
_polyfit.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
_polyfit.restype = None

def polyfit(fit, xv, yv, wv):
    "fit a poly to data"
    t0 = time.time()
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    if not (
            isinstance(xv, array.array) and \
            isinstance(yv, array.array) and \
            isinstance(wv, array.array)
        ):
        raise TypeError("bad types")
    if not (
            xv.typecode == 'd' and \
            yv.typecode == 'd' and \
            wv.typecode == 'd'
        ):
        raise TypeError("bad types")
    if len(xv) != polyfit_npoints(fit):
        raise ValueError("bad sizes")
    if not len(xv) == len(yv) == len(wv):
        raise ValueError("bad sizes")
    if min(wv) <= 0:
        raise ValueError("bad wv")
    xa, _ = xv.buffer_info()
    ya, _ = yv.buffer_info()
    wa, _ = wv.buffer_info()
    _polyfit(
        fit,
        ctypes.c_void_p(xa),
        ctypes.c_void_p(ya),
        ctypes.c_void_p(wa)
    )
    return time.time() - t0

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

_polyerr = _libpolyfit.polyfit_err
_polyerr.argtypes = [ctypes.c_void_p, ctypes.c_int]
_polyerr.restype  = ctypes.c_double


def polyfit_err(fit, degree):
    "return rms fit error"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    if not isinstance(degree, int):
        raise TypeError("bad degree type")
    if not 0 <= degree <= polyfit_maxdeg(fit):
        raise ValueError("illegal degree")
    return _polyerr(fit, degree)

_polyval = _libpolyfit.polyfit_val
_polyval.argtypes = [
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int
]
_polyval.restype = None

def polyfit_val(fit, x, degree, values):
    "evaluate poly and derivatives"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    if not isinstance(degree, int):
        raise TypeError("bad types")
    if not 0 <= degree <= polyfit_maxdeg(fit):
        raise ValueError("bad nderiv")
    if not isinstance(values, array.array):
        raise TypeError("bad type for values")
    if values.typecode != 'd':
        raise TypeError("bad type for values")
    if len(values) < 1:
        raise ValueError("bad values")
    da, _ = values.buffer_info()
    _polyval(fit, float(x), degree, ctypes.c_void_p(da), len(values) - 1)
    return values

_polycofs = _libpolyfit.polyfit_cofs
_polycofs.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_void_p
]
_polycofs.restype = None

def polyfit_cofs(fit, degree, x0, cofs):
    "return poly coefficients"
    if not isinstance(fit, int):
        raise TypeError("bad fit type")
    if not isinstance(degree, int):
        raise TypeError("bad degree type")
    if not 0 <= degree <= polyfit_maxdeg(fit):
        raise ValueError("bad degree")
    if not (isinstance(cofs, array.array) and cofs.typecode == 'd'):
        raise TypeError("bad cofs type")
    if len(cofs) < degree + 1:
        raise ValueError("bad cofs length")
    ca, _ = cofs.buffer_info()
    _polycofs(fit, degree, float(x0), ctypes.c_void_p(ca))
    return cofs
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
