/***********************************************************************
 * polyfit.h
 */

#ifndef polyfit_h__
#define polyfit_h__

#ifdef __cplusplus
extern "C" {
#endif

extern void *polyfit_alloc(
    const int npoints, const int maxdeg
);

extern void polyfit_free(
    void *fit
);

extern void polyfit(
    void * const fit,
    const double * const xv,
    const double * const yv,
    const double * const wv
);

extern void polyfit_val(
    const void * const fit,
    const double x,
    int degree,
    double * const derivatives,
    int nderiv
);

extern void polyfit_cofs(
    const void * const fit,
    const int degree,
    const double x0,
    double * const cofs
);

extern double polyfit_err(
    const void * const fit,
    const int degree
);

extern int polyfit_npoints(
    const void * const fit
);

extern int polyfit_maxdeg(
    const void * const fit
);

#ifdef __cplusplus
};
#endif

#endif

/** EOF */
