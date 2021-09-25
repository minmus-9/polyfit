/****************************************************************
 * polyfit.c - quad-precision orthogonal polynomial least squares
 *
 * there are no comments in this code. see the python code
 * for the commented reference implementation. this code
 * follows the python code exactly. see examples/ex2.c for
 * a demo.
 */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
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

#define PLAN_MAGIC 0x504c414e   /* "PLAN" */

typedef struct plan {
    uint32_t magic;     /* PLAN_MAGIC */
    int      D;         /* max fit degree */
    int      N;         /* data point count */
    Quad_t  *b;         /* maxdeg + 1 */
    Quad_t  *c;         /* maxdeg + 2 */
    Quad_t  *g;         /* maxdeg + 2 */
    Quad_t  *xv;        /* npoints */
    Quad_t  *wv;        /* npoints */
} plan_t;

typedef struct plan_scratch {
    Quad_t *phi_k;      /* npoints */
    Quad_t *phi_km1;    /* npoints */
    double *b_vec;      /* 2 * npoints */
    double *g_vec;      /* 2 * npoints */
} plan_scratch_t;

#define FIT_MAGIC 0x46495421    /* "FIT!" */

typedef struct fit {
    uint32_t      magic;     /* FIT_MAGIC */
    const plan_t *plan;
    Quad_t       *a;         /* maxdeg + 1 */
    double       *rms_errs;  /* maxdeg + 1 */
    double       *resids;    /* npoints */
} fit_t;

typedef struct fit_scratch {
    Quad_t  *phi_k;     /* npoints */
    Quad_t  *phi_km1;   /* npoints */
#define e_vec a_vec
    Quad_t  *rv;        /* npoints */
    double  *a_vec;     /* 2 * npoints */
} fit_scratch_t;

#define EVAL_MAGIC 0x4556414c   /* "EVAL" */

typedef struct eval {
    uint32_t     magic;     /* EVAL_MAGIC */
    const fit_t *fit;
    /* scratch, don't allocate for each call */
    Quad_t      *z_j;       /* maxdeg + 3 */
    Quad_t      *z_jm1;     /* maxdeg + 3 */
    double      *coefs;     /* maxdeg + 1 */
} eval_t;
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
static INLINE plan_t *alloc_plan(const int maxdeg, const int npoints) {
    plan_t *plan = calloc(1, sizeof(plan_t));

    if (plan == NULL)
        return NULL;
    plan->magic = PLAN_MAGIC;
    plan->D     = maxdeg;
    plan->N     = npoints;
#define N_PLAN_DATA (npoints * 2 + maxdeg * 3 + 5)
    if ((plan->b = malloc(N_PLAN_DATA * sizeof(Quad_t))) == NULL) {
        polyfit_free(plan);
        return NULL;
    }
    plan->c  = plan->b  + maxdeg + 1;
    plan->g  = plan->c  + maxdeg + 2;
    plan->xv = plan->g  + maxdeg + 2;
    plan->wv = plan->xv + npoints;

    plan->c[maxdeg + 1].x = plan->c[maxdeg + 1].xx = 0;
    plan->g[0].x  = 1;
    plan->g[0].xx = 0;
    return plan;
}

static INLINE void free_plan(plan_t * const plan) {
    free(plan->b);
    free(plan);
}

static INLINE void free_plan_scratch(plan_scratch_t * const scr);

static INLINE plan_scratch_t *alloc_plan_scratch(
    const plan_t * const plan
) {
    plan_scratch_t *scr = calloc(1, sizeof(plan_scratch_t));
    int npoints = plan->N;

    if (scr == NULL)
        return NULL;
    if ((scr->phi_k = malloc(2 * npoints * sizeof(Quad_t))) == NULL)
        goto bad;
    if ((scr->b_vec = malloc(4 * npoints * sizeof(double))) == NULL)
        goto bad;
    scr->phi_km1 = scr->phi_k + npoints;
    scr->g_vec   = scr->b_vec + 2 * npoints;
    return scr;

bad:
    free_plan_scratch(scr);
    return NULL;
}

static INLINE void free_plan_scratch(plan_scratch_t * const scr) {
    if (scr->phi_k != NULL)
        free(scr->phi_k);
    if (scr->b_vec != NULL)
        free(scr->b_vec);
    free(scr);
}

static INLINE fit_t *alloc_fit(const plan_t * const plan) {
    fit_t *fit = calloc(1, sizeof(fit_t));

    if (fit == NULL)
        return NULL;
    fit->magic = FIT_MAGIC;
    fit->plan  = plan;
#define N_FIT_A (plan->D + 1)
    if ((fit->a = malloc(N_FIT_A * sizeof(Quad_t))) == NULL)
        goto bad;
#define N_FIT_DBL (plan->N + plan->D + 1)
    if ((fit->rms_errs = malloc(N_FIT_DBL * sizeof(double))) == NULL)
        goto bad;
    fit->resids = fit->rms_errs + plan->D + 1;
    return fit;

bad:
    polyfit_free(fit);
    return NULL;
}

static INLINE void free_fit(fit_t *fit) {
    if (fit->a != NULL)
        free(fit->a);
    if (fit->rms_errs != NULL)
        free(fit->rms_errs);
    free(fit);
}

static INLINE void free_fit_scratch(fit_scratch_t *scr);

static INLINE fit_scratch_t *alloc_fit_scratch(fit_t *fit) {
    fit_scratch_t *scr = calloc(1, sizeof(fit_scratch_t));
    int npoints = fit->plan->N;

    if (scr == NULL)
        return NULL;
    if ((scr->phi_k = malloc(3 * npoints * sizeof(Quad_t))) == NULL)
        goto bad;
    if ((scr->a_vec = malloc(2 * npoints * sizeof(double))) == NULL)
        goto bad;
    scr->phi_km1 = scr->phi_k   + npoints;
    scr->rv      = scr->phi_km1 + npoints;
    return scr;

bad:
    free_fit_scratch(scr);
    return NULL;
}

static INLINE void free_fit_scratch(fit_scratch_t *scr) {
    if (scr->phi_k != NULL)
        free(scr->phi_k);
    if (scr->a_vec != NULL)
        free(scr->a_vec);
    free(scr);
}

static INLINE void free_eval(eval_t *ev);

static INLINE eval_t *alloc_eval(const fit_t * const fit) {
    eval_t *ev = calloc(1, sizeof(eval_t));
    int D = fit->plan->D;

    if (ev == NULL)
        return NULL;
    ev->magic = EVAL_MAGIC;
    ev->fit   = fit;
#define N_ZJ (3 * D + 7)
    if ((ev->z_j = malloc(N_ZJ * sizeof(Quad_t))) == NULL)
        goto bad;
    if ((ev->coefs = malloc((D + 1) * sizeof(double))) == NULL)
        goto bad;
    ev->z_jm1 = ev->z_j + D + 3;
    return ev;

bad:
    polyfit_free(ev);
    return NULL;
}

static INLINE void free_eval(eval_t *ev) {
    if (ev->z_j)
        free(ev->z_j);
    if (ev->coefs)
        free(ev->coefs);
    free(ev);
}

int polyfit_free(void * const polyfit_object) {
    plan_t *plan = polyfit_object;

    switch (plan->magic) {
        case PLAN_MAGIC:
            free_plan(plan);
            break;
        case FIT_MAGIC:
            free_fit((fit_t *) polyfit_object);
            break;
        case EVAL_MAGIC:
            free_eval((eval_t *) polyfit_object);
            break;
        default:
            errno = EINVAL;
            return -1;
    }
    return 0;
}

/* }}} */
/* {{{ polyfit_plan */
/**********************************************************************/
void *polyfit_plan(
        const int maxdeg,
        const double * const xv_,
        const double * const wv_,
        const int npoints
) {
    plan_t *plan;
    plan_scratch_t *w;

    int     i, k, bptr, gptr, N = npoints, D = maxdeg;
    Quad_t  b_k, c_k, g_k;
    Quad_t *xv, *wv, *phi_k, *phi_km1, *b, *c, *g;
    double *b_vec, *g_vec;

    if ((plan = alloc_plan(maxdeg, npoints)) == NULL)
        return NULL;
    if ((w = alloc_plan_scratch(plan)) == NULL) {
        polyfit_free(plan);
        return NULL;
    }
    xv      = plan->xv;
    wv      = plan->wv;
    phi_k   = w->phi_k;
    phi_km1 = w->phi_km1;
    b       = plan->b;
    c       = plan->c;
    g       = plan->g;
    b_vec   = w->b_vec;
    g_vec   = w->g_vec;

    for (i = 0; i < N; i++) {
        xv[i]      = to_quad(xv_[i]);
        wv[i]      = to_quad(wv_[i]);
        phi_k[i]   = one();
        phi_km1[i] = zero();
    }
    for (k = 0; k <= D; k++) {
        bptr = gptr = 0;
        for (i = 0; i < N; i++) {
            Quad_t s = mul(wv[i], phi_k[i]);
            Quad_t t = mul(s, phi_k[i]);

            vappend(b_vec, &bptr, mul(t, xv[i]));
            vappend(g_vec, &gptr, t);
        }
        g_k = vectorsum(g_vec, gptr);
        b_k = div_(vectorsum(b_vec, bptr), g_k);
        c_k = div_(g_k, g[k]);

        b[k]   = b_k;
        c[k]   = c_k;
        g[k+1] = g_k;

        if (k == D)
            continue;

        for (i = 0; i < N; i++) {
            Quad_t phi_kp1 = sub(
                mul(sub(xv[i], b_k), phi_k[i]), mul(c_k, phi_km1[i])
            );

            w->phi_km1[i] = w->phi_k[i];
            w->phi_k[i]   = phi_kp1;
        }
    }
    free_plan_scratch(w);
    return plan;
}
/* }}} */
/* {{{ polyfit_fit */
/**********************************************************************/
void *polyfit_fit(
        const void * const plan_,
        const double * const yv
) {
    const plan_t *plan = plan_;
    fit_t *fit;
    fit_scratch_t *w;

    int     i, k, aptr, N, D;
#define eptr aptr
    Quad_t  a_k;
    Quad_t *rv, *xv, *wv, *phi_k, *phi_km1, *a, *b, *c, *g;
    double *a_vec;
#define e_vec a_vec

    if ((plan == NULL) || (plan->magic != PLAN_MAGIC)) {
        errno = EINVAL;
        return NULL;
    }
    if ((fit = alloc_fit(plan)) == NULL)
        return NULL;
    if ((w = alloc_fit_scratch(fit)) == NULL) {
        polyfit_free(fit);
        return NULL;
    }
    N       = plan->N;
    D       = plan->D;
    xv      = plan->xv;
    wv      = plan->wv;
    phi_k   = w->phi_k;
    phi_km1 = w->phi_km1;
    a_vec   = w->a_vec;
    rv      = w->rv;
    a       = fit->a;
    b       = plan->b;
    c       = plan->c;
    g       = plan->g;

    for (i = 0; i < N; i++) {
        rv[i]      = to_quad(yv[i]);
        phi_k[i]   = one();
        phi_km1[i] = zero();
    }
    for (k = 0; k <= D; k++) {
        aptr = 0;
        for (i = 0; i < N; i++) {
            Quad_t s = mul(wv[i], phi_k[i]);

            vappend(a_vec, &aptr, mul(s, rv[i]));
        }
        a_k  = div_(vectorsum(a_vec, aptr), g[k + 1]);
        a[k] = a_k;

        eptr = 0;
        for (i = 0; i < N; i++) {
            rv[i] = sub(rv[i], mul(a_k, phi_k[i]));
            vappend(e_vec, &eptr, mul(rv[i], rv[i]));
        }
        fit->rms_errs[k] = sqrt(
            to_double(
                div_(vectorsum(e_vec, eptr), to_quad(N))
            )
        );

        if (k == D)
            continue;

        for (i = 0; i < N; i++) {
            Quad_t phi_kp1 = sub(
                mul(sub(xv[i], b[k]), phi_k[i]), mul(c[k], phi_km1[i])
            );

            w->phi_km1[i] = w->phi_k[i];
            w->phi_k[i]   = phi_kp1;
        }
    }
    for (i = 0; i < N; i++) {
        fit->resids[i] = to_double(rv[i]);
    }

    free_fit_scratch(w);
    return fit;
}
/* }}} */
/* {{{ polyfit_evaluator */
void *polyfit_evaluator(const void * const fit_) {
    const fit_t * const fit = fit_;

    if ((fit == NULL) || (fit->magic != FIT_MAGIC)) {
        errno = EINVAL;
        return NULL;
    }
    return alloc_eval(fit);
}
/* }}} */
/* {{{ polyfit_eval */
/**********************************************************************/

int polyfit_eval(
    void * const evaluator,
    const double x_,
    const int degree,
    double * const derivatives,
    const int nderiv
) {
    eval_t *ev = evaluator;
    Quad_t  x = to_quad(x_), fac = one(), val;
    Quad_t *b, *c, *z_j, *z_jm1;
    int     i, j, k, D, deriv_ptr = 0;
    int     d = degree, n = nderiv;

    if ((ev == NULL) || (ev->magic != EVAL_MAGIC)) {
        errno = EINVAL;
        return -1;
    }
    D     = ev->fit->plan->D;
    b     = ev->fit->plan->b;
    c     = ev->fit->plan->c;
    z_j   = ev->z_j;
    z_jm1 = ev->z_jm1;

    if (d < 0)
        d = D;
    else
        d = MIN(d, D);
    if (n < 0)
        n = d;

    for (i = 0; i <= d; i++) {
        z_jm1[i] = ev->fit->a[i];
    }
    for (i = 1; i <= 2; i++) {
        z_j[d + i] = zero();
    }

    for (j = 0; j <= MIN(d, n); j++) {
        if (j > 1) {
            fac = mul(fac, to_quad(j));
        }
        for (k = d; k >= j; k--) {
            int t = k - j;
            
            z_j[k] = add(
                z_jm1[k],
                sub(
                    mul(sub(x, b[t]), z_j[k + 1]),
                    mul(c[t+1], z_j[k + 2])
                )
            );
        }

        val = mul(fac, z_j[j]);
        derivatives[deriv_ptr++] = to_double(val);

        for (i = 0; i <= d + 2; i++) {
            z_jm1[i] = z_j[i];
        }
        for (i = 1; i <= 2; i++) {
            z_j[d + i] = zero();
        }
    }

    while (deriv_ptr < n)
        derivatives[deriv_ptr++] = 0;
    return 0;
}
/* }}} */
/* {{{ polyfit_coefs */
/**********************************************************************/
int polyfit_coefs(
    void * const evaluator,
    const double x0,
    const int degree,
    double * const coefs
) {
    eval_t *ev = evaluator;
    double  fac = 1;
    int     i, d = degree;

    if ((ev == NULL) || (ev->magic != EVAL_MAGIC)) {
        errno = EINVAL;
        return -1;
    }
    if (d < 0)
        d = ev->fit->plan->D;
    else
        d = MIN(d, ev->fit->plan->D);
    polyfit_eval(ev, x0, d, ev->coefs, d);
    for (i = 0; i <= d; i++) {
        if (i > 1)
            fac /= i;
        coefs[d - i] = fac * ev->coefs[i];
    }
    return 0;
}
/* }}} */
/* {{{ polyfit_npoints */
/**********************************************************************/
int polyfit_npoints(const void * const plan_) {
    const plan_t * const plan = plan_;

    return plan->N;
}
/* }}} */
/* {{{ polyfit_maxdeg */
/**********************************************************************/
int polyfit_maxdeg(const void * const plan_) {
    const plan_t * const plan = plan_;

    return plan->D;
}
/* }}} */
/* {{{ polyfit_resids */
const double *polyfit_resids(const void * const fit_) {
    const fit_t *fit = fit_;

    return fit->resids;
}
/* }}} */
/* {{{ polyfit_rms_errs */
extern const double *polyfit_rms_errs(const void * const fit_) {
    const fit_t *fit = fit_;

    return fit->rms_errs;
}
/* }}} */

/** EOF */
