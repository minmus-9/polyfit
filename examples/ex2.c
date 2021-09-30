/***************************************************************
 * ex2.c - polyfit demo
 */

#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include "polyfit.h"

#ifndef M_PI
#define M_PI 0
#define USE_ACOS
#endif

#define N 10000
#define D 3
double xv[N], yv[N], wv[N];

/* poly coefficients to fit, highest degree first */
double cv[D + 1] = { 2, 1, -1, M_PI };

void init() {
    double y;
    int    i, j;

#ifdef USE_ACOS
    cv[D] = acos(-1);
#endif
    for (i = 0; i < N; i++) {

        /* evaluate the poly to fit using horner's method */
        for (y = 0, j = 0; j <= D; j++) {
            y *= i;
            y += cv[j];
        }
        /* define xv[], yv[], and wv[] for the fit */
        xv[i] = i;
        yv[i] = y;
#if 1
        wv[i] = 1;
#else
        /* minimize relative residual */
        wv[i] = 1. / (y * y); /* y != 0 for this example poly */
#endif
    }
}

int main(int argc, char *argv[]) {
    void  *plan, *fit, *ev;
    int    i, j, n;
    double coefs[D + 1], d[D + 1];
    const double *t;

    /* fill in xv, yv, and, wv */
    init();

    /* create the fit plan */
    if ((plan = polyfit_plan(D, xv, wv, N)) == NULL) {
        perror("polyfit_plan");
        return 1;
    }

    /* compute the fit */
    if ((fit = polyfit_fit(plan, yv)) == NULL) {
        perror("polyfit_fit");
        return 1;
    }

    /* make an evaluator */
    if ((ev = polyfit_evaluator(fit)) == NULL) {
        perror("polyfit_evaluator");
        return 1;
    }

    /* print fit stats */
    if ((n = polyfit_maxdeg(plan)) < 0) {
        perror("polyfit_maxdeg");
        return 1;
    }
    printf("maxdeg %d\n", n);
    if ((n = polyfit_npoints(plan)) < 0) {
        perror("polyfit_npoints");
        return 1;
    }
    printf("points %d\n", n);

    /* print per-degree rms errors */
    if ((t = polyfit_rms_errs(fit, NULL)) == NULL) {
        perror("polyfit_rms_errs");
        return 1;
    }
    printf("erms  ");
    for (i = 0; i <= D; i++) {
        printf(" %.15e", t[i]);
    }
    printf("\n");

    /* print a few values */
    for (i = 0; i < 4; i++) {
        if (polyfit_eval(ev, xv[i], D, d, D) < 0) {
            perror("polyfit_eval");
            return 1;
        }
        printf("value  %.1f", xv[i]);
        for (j = 0; j <= D; j++) {
            printf(" %.15e", d[j]);
        }
        printf("\n");
    }

    /* print value and all derivatives for all degrees */
    for (i = 0; i <= D; i++) {
        if (polyfit_eval(ev, xv[0], i, d, -1) < 0) {
            perror("polyfit_eval");
            return 1;
        }
        printf("deg    %d", i);
        for (j = 0; j <= i; j++) {
            printf(" %.15e", d[j]);
        }
        printf("\n");
    }

    /* print coefficients for all degrees about (x - xv[0]) */
    for (i = 0; i <= D; i++) {
        if (polyfit_coefs(ev, xv[0], i, coefs) < 0) {
            perror("polyfit_coefs");
            return 1;
        }
        printf("coefs  %d", i);
        for (j = 0; j <= i; j++) {
            printf(" %.15e", coefs[j]);
        }
        printf("\n");
    }
    
    /* coefs halfway through */
    if (polyfit_coefs(ev, xv[N>>1], D, coefs) < 0) {
        perror("polyfit_coefs");
        return 1;
    }
    printf("coefs ");
    for (i = 0; i <= D; i++) {
        printf(" %.15e", coefs[i]);
    }
    printf("\n");

    /* free the fit objects */
    polyfit_free(ev);
    polyfit_free(fit);
    polyfit_free(plan);
    return 0;
}

/* EOF */
