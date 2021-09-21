/***********************************************************************
 * ex2.c - polyfit demo
 */

#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include "polyfit.h"

#define N 100000
#define D 3
double xv[N], yv[N], wv[N];

double cv[D + 1] = { 2, 1, -1, M_PI };

static void init() {
    double y;
    int    i, j;
    for (i = 0; i < N; i++) {

        for (y = 0, j = 0; j <= D; j++) {
            y *= i;
            y += cv[j];
        }
        xv[i] = i;
        yv[i] = y;
        wv[i] = 1 / (y * y);
    }
}

int main(int argc, char *argv[]) {
    void  *fit = polyfit_alloc(N, D);
    int    i;
    double cofs[D + 1], maxrel = -1, derivs[D + 1];

    struct timeval tv0, tv1;

    if (fit == NULL) {
        perror("calloc");
        return 1;
    }

    init();

    gettimeofday(&tv0, NULL);
    polyfit(fit, xv, yv, wv);
    gettimeofday(&tv1, NULL);
    printf("dt = %f\n",
        (float) (
            (tv1.tv_sec - tv0.tv_sec) + 1e-6 * (tv1.tv_usec - tv0.tv_usec)
        )
    );

    printf("preds\n");
    for (i = 0; i < 10; i++) {
        double diff, d[1] = { 0 };

        polyfit_val(fit, xv[i], D, d, 0);
        diff = yv[i] - d[0];
        printf(
            "  %f %.3e %.3e %.3e\n",
            (float) xv[i], (float) yv[i], (float) d[0], (float) diff);
    }

    printf("derivs x=0\n");
    polyfit_val(fit, xv[0], D, derivs, D);
    for (i = 0; i <= D; i++) {
        printf("  %d %.3e\n", i, (float) derivs[i]);
    }

    printf("derivs x=1\n");
    polyfit_val(fit, xv[1], D, derivs, D);
    for (i = 0; i <= D; i++) {
        printf("  %d %.3e\n", i, (float) derivs[i]);
    }

    polyfit_cofs(fit, D, 0, cofs);
    printf("cofs\n");
    for (i = 0; i <= D; i++) {
        printf("  %d %.3e\n", i, (float) cofs[i]);
    }

    printf("errs\n");
    for (i = 0; i <= D; i++) {
        printf("  %d %.3e\n", i, (float) polyfit_err(fit, i));
    }

    for (i = 0; i < N; i++) {
        double err, d[1] = { 0 };

        polyfit_val(fit, xv[i], D, d, 0);
        err = fabs(d[0] / yv[i] - 1);
        if (err > maxrel)
            maxrel = err;
    }
    printf("maxrel %.3e\n", (float) maxrel);

    polyfit_free(fit);
    return 0;
}

/* EOF */
