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
    void  *plan, *fit;
    int    i, j;
    double cofs[D + 1], d[D + 1], *t;

    /* fill in xv, yv, and, wv */
    init();

    /* create the fit plan */
    if ((plan = polyfit_plan(D, xv, wv, N)) == NULL) {
        perror("calloc");
        return 1;
    }

    /* compute the fit */
    if ((fit = polyfit_fit(plan, yv)) == NULL) {
        perror("calloc");
        return 1;
    }

    /* print fit stats */
    printf("maxdeg %d\n", polyfit_maxdeg(fit));
    printf("points %d\n", polyfit_npoints(fit));

    /* print per-degree rms errors */
    t = polyfit_rms_errs(fit);
    printf("erms  ");
    for (i = 0; i <= D; i++) {
        printf(" %.18e", t[i]);
    }
    printf("\n");

    /* print a few values */
    for (i = 0; i < 4; i++) {
        polyfit_eval(fit, xv[i], D, d, D);
        printf("value  %f", xv[i]);
        for (j = 0; j <= D; j++) {
            printf(" %.18e", d[j]);
        }
        printf("\n");
    }

    /* print value and all derivatives for all degrees */
    for (i = 0; i <= D; i++) {
        polyfit_eval(fit, xv[0], i, d, -1);
        printf("deg    %d", i);
        for (j = 0; j <= i; j++) {
            printf(" %.18e", d[j]);
        }
        printf("\n");
    }

    /* print coefficients for all degrees about (x - xv[0]) */
    for (i = 0; i <= D; i++) {
        polyfit_cofs(fit, xv[0], i, cofs);
        printf("coefs  %d", i);
        for (j = 0; j <= i; j++) {
            printf(" %.18e", cofs[j]);
        }
    }
    
    /* coefs halfway through */
    polyfit_cofs(fit, D, xv[N>>1], cofs);
    printf("coefs ");
    for (i = 0; i <= D; i++) {
        printf(" %.18e", cofs[i]);
    }
    printf("\n");

    /* free the fit objects */
    polyfit_free(fit);
    polyfit_free(plan);
    return 0;
}

/* EOF */
