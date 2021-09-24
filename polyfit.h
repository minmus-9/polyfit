/***********************************************************************
 * polyfit.h
 */

#ifndef polyfit_h__
#define polyfit_h__

#ifdef __cplusplus
extern "C" {
#endif

extern void polyfit_free(
    void *plan_or_fit
);

extern void *polyfit_plan(
    const int maxdeg,
    const double * const xv,
    const double * const wv,
    int npoints
);

extern void *polyfit_fit(
    const void * const plan,
    const double * const yv,
);

extern void polyfit_eval(
    const void * const fit,
    const double x,
    int degree,
    double * const derivatives,
    int nderiv
);

extern void polyfit_coefs(
    const void * const fit,
    const double x0,
    const int degree,
    double * const coefs
);

extern int polyfit_maxdeg(
    const void * const fit
);

extern int polyfit_npoints(
    const void * const fit
);

extern const double *polyfit_resids(
    const void * const fit
);

extern const double *polyfit_rms_errs(
    const void * const fit,
);

#ifdef __cplusplus
};
#endif

#endif

/** EOF */
