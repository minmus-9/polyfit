/***********************************************************************
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

#define N 100000
#define D 3
double xv[N], yv[N], wv[N];

/* poly to fit, highest coef first */
double cv[D + 1] = { 2, 1, -1, M_PI };

void init() {
    double y;
    int    i, j;

#ifdef USE_ACOS
    cv[D] = acos(-1);
#endif
    for (i = 0; i < N; i++) {

        /* evaluate poly to fit using horner's method */
        for (y = 0, j = 0; j <= D; j++) {
            y *= i;
            y += cv[j];
        }
        /* define xv[], yv[], and wv[] for the fit */
        xv[i] = i;
        yv[i] = y;
        /* minimize relative residual */
        wv[i] = 1 / (y * y);    /* NB y != 0 for this example poly */
    }
}

int main(int argc, char *argv[]) {
    void  *fit;
    int    i, j;
    double cofs[D + 1], maxrel = -1, d[D + 1];

    struct timeval tv0, tv1;

    /* allocate the workspace */
    if ((fit = polyfit_alloc(N, D)) == NULL) {
        perror("calloc");
        return 1;
    }

    /* fill in xv, yv, and, wv */
    init();

    /* do the fit */
    gettimeofday(&tv0, NULL);
    polyfit(fit, xv, yv, wv);
    gettimeofday(&tv1, NULL);

    /* print fit stats */
    printf("maxdeg %d\n", polyfit_maxdeg(fit));
    printf("points %d\n", polyfit_npoints(fit));
    printf("time   %f\n",
        (float) (
            (tv1.tv_sec - tv0.tv_sec) + 1e-6 * (tv1.tv_usec - tv0.tv_usec)
        )
    );

    /* print per-degree rms errors */
    printf("erms  ");
    for (i = 0; i <= D; i++) {
        printf(" %.3e", polyfit_err(fit, i));
    }
    printf("\n");

    /* print relative resid error across all xv_i */
    for (i = 0; i < N; i++) {
        double err;

        polyfit_val(fit, xv[i], D, d, 0);
        err = fabs(d[0] / yv[i] - 1);
        if (err > maxrel)
            maxrel = err;
    }
    printf("relerr %.3e\n", (float) maxrel);

    /* print some values */
    for (i = 0; i < 5; i++) {
        polyfit_val(fit, xv[i], D, d, D);
        printf("value  %f", (float) xv[i]);
        for (j = 0; j <= D; j++) {
            printf(" %.3e", (float) d[j]);
        }
        printf("\n");
    }

    /* coefs about x0=0 */
    polyfit_cofs(fit, D, xv[0], cofs);
    printf("coefs0");
    for (i = 0; i <= D; i++) {
        printf(" %.3e", (float) cofs[i]);
    }
    printf("\n");

    /* value and all derivs at xv[0] */
    printf("value0");
    polyfit_val(fit, xv[0], D, d, D);
    for (j = 0; j <= D; j++) {
        printf(" %.3e", (float) d[j]);
    }
    printf("\n");
    
    /* free the workspace */
    polyfit_free(fit);
    return 0;
}

/* EOF */
