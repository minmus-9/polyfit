# polyfit
double-double precision orthogonal polynomial least squares

copy of the abstract from polyfit.pdf:

In this note I present the Python2, Python3, and C package
that implements double-double precision least-squares polynomial
fitting using orthogonal polynomials. Pure Python and C
versions are provided; they are slower than the traditional
approach (primarily due to being double-double precision), but
are numerically more stable and accurate than the traditional
approach. A listing of the source code for the 264 SLOC Python
reference implementation is included in appendix. A much faster
C version also ships with this package; there is a Python
ctypes-based interface called cpolyfit that integrates this fast
version into Python. Both Python interfaces support Python 2.7
and 3.6+.

please see ployfit.pdf for further details.
