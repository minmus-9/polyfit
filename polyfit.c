/***********************************************************************
 * polyfit.c
 *
 * there are no comments in this code. see the python code
 * for the commented reference implementation. this code
 * follows the python code exactly. see examples/ex2.c for
 * a demo.
 */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "polyfit.h"

#ifdef __STRICT_ANSI__
#define INLINE
#else
#define INLINE inline
#endif

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))

/* {{{ definitions */
typedef struct Quad {
    double x, xx;
} Quad_t;

typedef struct work {
    int a_vec_ptr, b_vec_ptr, g_vec_ptr, e_vec_ptr;

#define a_vec work1_d2n
#define cof_vec work1_d2n
    double *work1_d2n;  /* npoints * 2 */

#define b_vec work2_d2n
    double *work2_d2n;  /* npoints * 2 */

#define g_vec work3_d2n
    double *work3_d2n;  /* npoints * 2 */

#define e_vec work4_d2n
    double *work4_d2n;  /* npoints * 2 */

#define x_vec work1_qn
    Quad_t *work1_qn;   /* npoints */

#define r_vec work2_qn
    Quad_t *work2_qn;   /* npoints */

#define w_vec work3_qn
    Quad_t *work3_qn;   /* npoints */

#define phi_k work4_qn
#define z_j   work4_qn
    Quad_t *work4_qn;   /* npoints */

#define phi_km1 work5_qn
#define z_jm1   work5_qn
    Quad_t *work5_qn;   /* npoints */

#define g work1_qdp2    /* maxdeg + 2 */
    Quad_t *work1_qdp2;
} work_t;

typedef struct fit {
    int     npoints, maxdeg;
    work_t *w;
    Quad_t *a;  /* maxdeg + 1 */
    Quad_t *b;  /* maxdeg + 1 */
    Quad_t *c;  /* maxdeg + 2 */
    double *e;  /* maxdeg + 1 */
} fit_t;
/* }}} */
/* {{{  quad-precision functions from ogita and dekker */
/**********************************************************************/
static INLINE Quad_t twosum(const double a, const double b) {
    double x, y, z;
    Quad_t q;

    x = a + b;
    z = x - a;
    y = (a - (x - z)) + (b - z);

    q.x  = x;
    q.xx = y;
    return q;
}

static INLINE Quad_t twodiff(const double a, const double b) {
    double x, y, z;
    Quad_t q;

    x = a - b;
    z = x - a;
    y = (a - (x - z)) - (b + z);

    q.x  = x;
    q.xx = y;
    return q;
}

#define FACTOR ((double) (1 + (1 << 27)))

static INLINE Quad_t split(const double a) {
    double c, x, y;
    Quad_t q;

    c = FACTOR * a;
    x = c - (c - a);
    y = a - x;

    q.x  = x;
    q.xx = y;
    return q;
}

static INLINE Quad_t twoproduct(const double a, const double b) {
    double x, y;
    Quad_t aa, bb;

    x  = a * b;
    aa = split(a);
    bb = split(b);
    y  = aa.xx * bb.xx -
        (x - aa.x * bb.x - aa.xx * bb.x - aa.x * bb.xx);
    return twosum(x, y);
}

static INLINE Quad_t sum2s(const double * const p, const int np) {
    double sigma;
    Quad_t q;
    int    i;

    if (!np) {
        q.x = q.xx = 0;
        return q;
    }

    q.x   = p[0];
    q.xx  = 0;
    sigma = 0;
    for (i = 1; i < np; i++) {
        q      = twosum(q.x, p[i]);
        sigma += q.xx;
    }
    return twosum(q.x, sigma);
}

static INLINE void vsum(double * const p, const int np) {
    Quad_t q;
    int    i, im1 = 0;

    for (i = 1; i < np; i++) {
        q      = twosum(p[i], p[im1]);
        p[i]   = q.x;
        p[im1] = q.xx;
        im1    = i;
    }
}

static INLINE void sumkcore(
    double * const p, const int np, const int K
) {
    int i;

    for (i = 1; i < K; i++) {
        vsum(p, np);
    }
}

static INLINE Quad_t sumk(
    double * const p, const int np, const int K
) {
    sumkcore(p, np, K);
    return sum2s(p, np);
}

static INLINE Quad_t vectorsum(double * const p, const int np) {
    return sumk(p, np, 3);
}
/* }}} */
/* {{{ utility functions */
/**********************************************************************/
static INLINE void vappend(
    double * const vec,
    int * const vecptr,
    const Quad_t src
) {
    int v = *vecptr;

    vec[v++] = src.x;
    vec[v++] = src.xx;
    *vecptr  = v;
}

static INLINE Quad_t zero() {
    Quad_t q = { 0, 0 };

    return q;
}

static INLINE Quad_t one() {
    Quad_t q = { 1, 0 };

    return q;
}

static INLINE Quad_t to_quad(const double x) {
    Quad_t q = { 0, 0 };
    
    q.x = x;
    return q;
}

static INLINE double to_double(const Quad_t q) {
    return q.x;
}
/* }}} */
/* {{{ quad-precision arithmetic */
/**********************************************************************/
static INLINE Quad_t add(const Quad_t x, const Quad_t y) {
    Quad_t z = twosum(x.x, y.x);

    return twosum(z.x, z.xx + x.xx + y.xx);
}

static INLINE Quad_t sub(const Quad_t x, const Quad_t y) {
    Quad_t z = twodiff(x.x, y.x);

    return twosum(z.x, z.xx + x.xx - y.xx);
}

static INLINE Quad_t mul(const Quad_t x, const Quad_t y) {
    Quad_t z = twoproduct(x.x, y.x);

    z.xx += x.xx * y.x + x.x * y.xx;
    return twosum(z.x, z.xx);
}

static INLINE Quad_t div_(const Quad_t x, const Quad_t y) {
    double c = x.x / y.x;
    double cc;
    Quad_t u = twoproduct(c, y.x);
    
    cc = (x.x - u.x - u.xx + x.xx - c * y.xx) / y.x;

    return twosum(c, cc);
}

static INLINE Quad_t sqrt_(const Quad_t x) {
    double c, cc;
    Quad_t u;

    if (!(x.x || x.xx)) return zero();
    c  = sqrt(x.x);
    u  = twoproduct(c, c);
    cc = (x.x - u.x - u.xx + x.xx) * 0.5 / c;
    return twosum(c, cc);
}
/* }}} */
/* {{{ memory allocation */
/**********************************************************************/
void *polyfit_alloc(
    const int npoints, const int maxdeg
) {
    fit_t  *fit;
    work_t *work;
    int     alloc45 = MAX(npoints, maxdeg + 3);

    errno = 0;
    if (maxdeg < 0) return NULL;
    if (npoints < maxdeg + 1) return NULL;

    if ((fit = calloc(1, sizeof(fit_t))) == NULL) return NULL;

    if ((fit->w = work = calloc(1, sizeof(work_t))) == NULL)
        goto bad;

    fit->npoints = npoints;
    fit->maxdeg  = maxdeg;

    if ((fit->a = calloc(maxdeg + 1, sizeof(Quad_t))) == NULL)
        goto bad;
    if ((fit->b = calloc(maxdeg + 1, sizeof(Quad_t))) == NULL)
        goto bad;
    if ((fit->c = calloc(maxdeg + 2, sizeof(Quad_t))) == NULL)
        goto bad;
    if ((fit->e = calloc(maxdeg + 1, sizeof(Quad_t))) == NULL)
        goto bad;

    if ((work->work1_d2n = calloc(2 * npoints, sizeof(double))) == NULL)
        goto bad;
    if ((work->work2_d2n = calloc(2 * npoints, sizeof(double))) == NULL)
        goto bad;
    if ((work->work3_d2n = calloc(2 * npoints, sizeof(double))) == NULL)
        goto bad;
    if ((work->work4_d2n = calloc(2 * npoints, sizeof(double))) == NULL)
        goto bad;

    if ((work->work1_qn = calloc(npoints, sizeof(Quad_t))) == NULL)
        goto bad;
    if ((work->work2_qn = calloc(npoints, sizeof(Quad_t))) == NULL)
        goto bad;
    if ((work->work3_qn = calloc(npoints, sizeof(Quad_t))) == NULL)
        goto bad;
    if ((work->work4_qn = calloc(alloc45, sizeof(Quad_t))) == NULL)
        goto bad;
    if ((work->work5_qn = calloc(alloc45, sizeof(Quad_t))) == NULL)
        goto bad;

    if ((work->work1_qdp2 = calloc(maxdeg + 2, sizeof(Quad_t))) == NULL)
        goto bad;

    return fit;

bad:
    polyfit_free(fit);
    return NULL;
}

/**********************************************************************/
#define FREE(p) do { if (p) { free(p); p = NULL; } } while (0)

void polyfit_free(
    void *fit
) {
    fit_t  *f = fit;
    work_t *work = f->w;

    if (work != NULL) {
        FREE(work->work1_d2n);
        FREE(work->work2_d2n);
        FREE(work->work3_d2n);
        FREE(work->work4_d2n);
        FREE(work->work1_qn);
        FREE(work->work2_qn);
        FREE(work->work3_qn);
        FREE(work->work4_qn);
        FREE(work->work5_qn);
        FREE(work->work1_qdp2);
    }
    FREE(work);
    FREE(f->a);
    FREE(f->b);
    FREE(f->c);
    FREE(f->e);
    FREE(f);
}
/* }}} */
/* {{{ polyfit */
/**********************************************************************/
void polyfit(
        void * const fit,
        const double * const xv,
        const double * const yv,
        const double * const wv
) {
    fit_t  *f = fit;
    work_t *w = f->w;
    int     i, k, N = f->npoints, D = f->maxdeg;
    Quad_t  a_k, b_k, c_k, g_k;

    for (i = 0; i < N; i++) {
        w->x_vec[i]   = to_quad(xv[i]);
        w->r_vec[i]   = to_quad(yv[i]);
        w->w_vec[i]   = to_quad(wv[i]);
        w->phi_k[i]   = one();
        w->phi_km1[i] = zero();
    }
    w->g[0] = one();
    f->c[D + 1] = zero();
    for (k = 0; k <= D; k++) {
        w->a_vec_ptr = w->b_vec_ptr = w->g_vec_ptr = 0;
        for (i = 0; i < N; i++) {
            Quad_t s = mul(w->w_vec[i], w->phi_k[i]);
            Quad_t t = mul(s, w->phi_k[i]);

            vappend(w->a_vec, &w->a_vec_ptr, mul(s, w->r_vec[i]));
            vappend(w->b_vec, &w->b_vec_ptr, mul(t, w->x_vec[i]));
            vappend(w->g_vec, &w->g_vec_ptr, t);
        }
        g_k = vectorsum(w->g_vec, w->g_vec_ptr);
        a_k = div_(vectorsum(w->a_vec, w->a_vec_ptr), g_k);
        b_k = div_(vectorsum(w->b_vec, w->b_vec_ptr), g_k);
        c_k = div_(g_k, w->g[k]);

        f->a[k]   = a_k;
        f->b[k]   = b_k;
        f->c[k]   = c_k;
        w->g[k+1] = g_k;

        for (i = 0; i < N; i++) {
            w->r_vec[i] = sub(w->r_vec[i], mul(a_k, w->phi_k[i]));
        }

        w->e_vec_ptr = 0;
        for (i = 0; i < N; i++) {
            vappend(
                w->e_vec, &w->e_vec_ptr, mul(w->r_vec[i], w->r_vec[i])
            );
        }
        f->e[k] = to_double(
            sqrt_(div_(vectorsum(w->e_vec, w->e_vec_ptr), to_quad(N)))
        );

        if (k == D)
            continue;

        for (i = 0; i < N; i++) {
            Quad_t phi_kp1 = sub(
                mul(sub(w->x_vec[i], b_k), w->phi_k[i]),
                mul(c_k, w->phi_km1[i])
            );

            w->phi_km1[i] = w->phi_k[i];
            w->phi_k[i]   = phi_kp1;
        }
    }
}
/* }}} */
/* {{{ polyfit_val */
/**********************************************************************/

void polyfit_val(
    const void * const fit,
    const double x_,
    int degree,
    double * const derivatives,
    int nderiv
) {
    const fit_t * const f = fit;
    work_t *w = f->w;
    Quad_t  x = to_quad(x_), fac = one(), val;
    int     i, j, k, maxdeg = f->maxdeg, deriv_ptr = 0;

    if (degree < 0)
        degree = maxdeg;
    if (nderiv < 0)
        nderiv = degree;

    for (i = 0; i <= degree; i++) {
        w->z_jm1[i] = f->a[i];
    }
    for (i = 1; i <= 2; i++) {
        w->z_j[degree + i] = zero();
    }

    for (j = 0; j <= MIN(degree, nderiv); j++) {
        if (j > 1) {
            fac = mul(fac, to_quad(j));
        }
        for (k = degree; k >= j; k--) {
            int    t = k - j;
            Quad_t tmp = sub(
                mul(sub(x, f->b[t]), w->z_j[k + 1]),
                mul(f->c[t+1], w->z_j[k + 2])
            );
            
            w->z_j[k] = add(w->z_jm1[k], tmp);
        }

        val = mul(fac, w->z_j[j]);
        derivatives[deriv_ptr++] = to_double(val);

        for (i = 0; i <= degree + 2; i++) {
            w->z_jm1[i] = w->z_j[i];
        }
        for (i = 1; i <= 2; i++) {
            w->z_j[degree + i] = zero();
        }
    }

    while (nderiv > degree) {
        derivatives[deriv_ptr++] = 0;
        nderiv--;
    }
}
/* }}} */
/* {{{ polyfit_cofs */
/**********************************************************************/
void polyfit_cofs(
    const void * const fit,
    const int degree,
    const double x0,
    double * const cofs
) {
    const   fit_t * const f = fit;
    work_t *w = f->w;
    double  fac = 1;
    int     i;

    polyfit_val(fit, x0, degree, w->cof_vec, degree);
    for (i = 1; i <= degree; i++) {
        fac /= i;
        cofs[degree - i] = fac * w->cof_vec[i];
    }
    cofs[degree] = w->cof_vec[0];
}
/* }}} */
/* {{{ polyfit_err */
/**********************************************************************/
double polyfit_err(
    const void * const fit,
    const int degree
) {
    const fit_t * const f = fit;

    return f->e[degree];
}
/* }}} */
/* {{{ polyfit_npoints */
/**********************************************************************/
int polyfit_npoints(
    const void * const fit
) {
    const fit_t * const f = fit;

    return f->npoints;
}
/* }}} */
/* {{{ polyfit_maxdeg */
/**********************************************************************/
int polyfit_maxdeg(
    const void * const fit
) {
    const fit_t * const f = fit;

    return f->maxdeg;
}
/* }}} */
/* {{{ polyfit_dump */
void polyfit_dump(
    const void * const fit
) {
    const fit_t *f = fit;
    int i, d = f->maxdeg;

    printf("a\n");
    for (i = 0; i <= d; i++) {
        printf("%2d %.18e %.18e\n", i, f->a[i].x, f->a[i].xx);
    }
    printf("b\n");
    for (i = 0; i <= d; i++) {
        printf("%2d %.18e %.18e\n", i, f->b[i].x, f->b[i].xx);
    }
    printf("c\n");
    for (i = 0; i <= d; i++) {
        printf("%2d %.18e %.18e\n", i, f->c[i].x, f->c[i].xx);
    }
    printf("e\n");
    for (i = 0; i <= d; i++) {
        printf("%2d %.18e\n", i, f->e[i]);
    }
}
/* }}} */

/** EOF */
