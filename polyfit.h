/***********************************************************************
 * polyfit.h
 */

#ifndef polyfit_h__
#define polyfit_h__

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************
 * given x values in xv[] and positive weights in wv[],
 * make a plan to perform least squares fitting up to
 * degree maxdeg and return a plan object. returns NULL
 * and sets errno on error.
 */
extern void *polyfit_plan(
    const int maxdeg,
    const double * const xv,
    const double * const wv,
    const int npoints
);

/***************************************************************
 * given a set of y values in yv[], compute all least
 * squares fits to yv[] up to degree maxdeg and return
 * a fit object. returns NULL and sets errno on error.
 */
extern void *polyfit_fit(
    const void * const plan,
    const double * const yv,
);

/***************************************************************
 * given a fit, return an evaluator that can (a) compute the
 * fit polynomial and its derivatives and (b) can compute
 * coefficients of the polynomial about a given point x0.
 * returns NULL and sets errno on error.
 */
extern void *polyfit_evaluator(
    const void * const fit
);

/***************************************************************
 * given a point x, a least squares fit degree degree,
 * and a desired number of derivatives to compute nderiv,
 * calculate and return the value of the polynomial and
 * any requested derivatives.
 * 
 * if degree is negative, use maxdeg instead. if nderiv is
 * negative, use the final value of deg; otherwise, compute
 * nderiv derivatives of the least squares polynomial of
 * degree deg.
 * 
 * the derivatives array contains the polynomial value first,
 * followed by any requested derivatives.
 *
 * returns 0 on success. on failure returns -1 and sets errno:
 *   EINVAL - evaluator is not an evaluator.
 *          - derivatives is NULL
 */
extern int polyfit_eval(
    void * const evaluator,
    const double x,
    const int degree,
    double * const derivatives,
    const int nderiv
);

/***************************************************************
 * return the coefficients of the fit polynomial of degree
 * degree about (x - x0). if degree is negative, use maxdeg
 * instead.
 *
 * returns 0 on success. on failure returns -1 and sets errno:
 *   EINVAL - evaluator is not an evaluator.
 *          - coefs is NULL
 */
extern int polyfit_coefs(
    const void * const evaluator,
    const double x0,
    const int degree,
    double * const coefs
);

/***************************************************************
 * return the maximum fit degree
 *
 * returns 0 on success. on failure returns -1 and sets errno:
 *   EINVAL - plan is not a plan.
 */
extern int polyfit_maxdeg(
    const void * const plan
);

/***************************************************************
 * return the number of fit points
 *
 * returns 0 on success. on failure returns -1 and sets errno:
 *   EINVAL - plan is not a plan.
 */
extern int polyfit_npoints(
    const void * const plan
);

/***************************************************************
 * return the list of residuals for the maxdeg fit.
 *
 * returns 0 on success. on failure returns -1 and sets errno:
 *   EINVAL - fit is not a fit.
 */
extern const double *polyfit_resids(
    const void * const fit
);

/***************************************************************
 * return a list of rms errors, one per fit degree. use them
 * to detect overfitting.
 *
 * returns 0 on success. on failure returns -1 and sets errno:
 *   EINVAL - fit is not a fit.
 */
extern const double *polyfit_rms_errs(
    const void * const fit,
);

/***************************************************************
 * free a plan, fit, or evaluator object
 *
 * returns 0 on success. on failure returns -1 and sets errno:
 *   EINVAL - unrecognized object type.
 */
extern int polyfit_free(
    void * const polyfit_object
);

#ifdef __cplusplus
};
#endif

#endif

/** EOF */
