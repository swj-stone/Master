/* Unit-ridge, held-loss minPI on disjoint two-column blocks. */
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API
#endif

enum { CAP = 192 };

typedef struct {
    int n, count[2], both;
    double mean[2], square[2], cross, shared_sum[2];
    double *rhs[2], *observed_y[2];
} Half;

static inline double load_x(const char *ptr, int size) {
    if (size == 4) {
        float x;
        memcpy(&x, ptr, sizeof(x));
        return x;
    }
    double x;
    memcpy(&x, ptr, sizeof(x));
    return x;
}

static void read_half(Half *h, const char *x, int size, int64_t rs, int64_t cs,
                      const int64_t *rows, int n, int64_t j, int64_t mate,
                      const double *target, int classification, int nc,
                      const double *target_mean, double *scratch) {
    double x0[CAP], x1[CAP], anchor[2] = {0., 0.}, offset[2] = {0., 0.};
    double sums[2] = {0., 0.};
    h->n = n;
    for (int k = 0; k < 2; ++k) {
        h->rhs[k] = scratch + (2*k)*nc;
        h->observed_y[k] = scratch + (2*k+1)*nc;
    }
    for (int i = 0; i < n; ++i) {
        const char *row = x + rows[i]*rs;
        double v[2] = {load_x(row + j*cs, size), load_x(row + mate*cs, size)};
        for (int k = 0; k < 2; ++k) {
            if (isfinite(v[k])) {
                if (h->count[k]++ == 0) anchor[k] = v[k];
                v[k] -= anchor[k];
                offset[k] += v[k];
            }
        }
        x0[i] = v[0];
        x1[i] = v[1];
    }
    for (int k = 0; k < 2; ++k) {
        offset[k] /= h->count[k] ? h->count[k] : 1;
        h->mean[k] = anchor[k] + offset[k];
    }
    int complete = h->count[0] == n && h->count[1] == n;
    for (int i = 0; i < n; ++i) {
        int seen0 = complete || isfinite(x0[i]);
        int seen1 = complete || isfinite(x1[i]);
        double a = seen0 ? x0[i] - offset[0] : 0.;
        double b = seen1 ? x1[i] - offset[1] : 0.;
        h->square[0] += a*a;
        h->square[1] += b*b;
        h->cross += a*b;
        int t = classification ? (int)target[i] : 0;
        double weight = classification ? 1. : target[i] - target_mean[0];
        h->rhs[0][t] += a*weight;
        h->rhs[1][t] += b*weight;
        sums[0] += a;
        sums[1] += b;
        if (!complete) {
            if (seen0) h->observed_y[0][t] += weight;
            if (seen1) h->observed_y[1][t] += weight;
            if (seen0 && seen1) {
                h->shared_sum[0] += a;
                h->shared_sum[1] += b;
                ++h->both;
            }
        }
    }
    if (complete) h->both = n;
    if (classification) {
        for (int k = 0; k < 2; ++k) {
            for (int t = 0; t < nc; ++t) {
                h->rhs[k][t] -= sums[k]*target_mean[t];
                if (!complete) h->observed_y[k][t] -= h->count[k]*target_mean[t];
            }
        }
    }
}

static void evaluate(const Half *fit, const Half *held, const double *ym_fit,
                     const double *ym_held, int nc, int64_t p, int64_t j,
                     int64_t mate, double *out) {
    double root[2], dx[2], norm[2], alone[2] = {0., 0.}, part[2] = {0., 0.};
    for (int k = 0; k < 2; ++k) {
        root[k] = fit->square[k] > 0. ? sqrt(fit->square[k]*(1.+1./fit->n)) : 1.;
        dx[k] = held->mean[k] - fit->mean[k];
        norm[k] = (held->square[k] + held->count[k]*dx[k]*dx[k])/root[k]/root[k];
    }
    double rho = fit->cross/root[0]/root[1], determinant = 1.-rho*rho;
    double cross = (held->cross + held->shared_sum[0]*dx[1] + held->shared_sum[1]*dx[0]
                    + held->both*dx[0]*dx[1])/root[0]/root[1];
    double product = 0.;
    for (int t = 0; t < nc; ++t) {
        double single[2] = {fit->rhs[0][t]/root[0], fit->rhs[1][t]/root[1]};
        double pair[2] = {(single[0]-rho*single[1])/determinant,
                          (single[1]-rho*single[0])/determinant};
        product += pair[0]*pair[1];
        double dy = ym_held[t]-ym_fit[t];
        for (int k = 0; k < 2; ++k) {
            double rhs = (held->rhs[k][t] + dx[k]*(held->observed_y[k][t] + held->count[k]*dy))/root[k];
            alone[k] += 2.*rhs*single[k] - norm[k]*single[k]*single[k];
            part[k] += 2.*rhs*pair[k] - norm[k]*pair[k]*pair[k];
        }
    }
    double joint = part[0] + part[1] - 2.*cross*product;
    out[j] = alone[0]/held->n;
    out[p+j] = mate == j ? out[j] : (joint-alone[1])/held->n;
    if (mate != j) {
        out[mate] = alone[1]/held->n;
        out[p+mate] = (joint-alone[0])/held->n;
    }
}

API int pivotal_endpoints(const char *x, int size, int64_t rs, int64_t cs, int64_t p,
                          const int64_t *rows, int nl, int nr, const int64_t *mates,
                          const double *target, int nc, int classification, int directions,
                          double *out) {
    if (nl < 1 || nr < 1 || nl + nr > CAP || nc < 1 || nc > CAP || (size != 4 && size != 8))
        return 1;
    double means[2][CAP] = {{0.}}, scratch[8*CAP];
    for (int side = 0; side < 2; ++side) {
        int start = side == 0 ? 0 : nl, n = side == 0 ? nl : nr;
        double base = classification ? 0. : target[start];
        for (int i = start; i < start+n; ++i) {
            if (classification) {
                int t = (int)target[i];
                if (t < 0 || t >= nc) return 2;
                means[side][t] += 1.;
            } else {
                means[side][0] += target[i]-base;
            }
        }
        for (int t = 0; t < nc; ++t) means[side][t] = base + means[side][t]/n;
    }
    for (int64_t j = 0; j < p; ++j) {
        int64_t k = mates[j];
        if (k < j) continue;
        Half left = {0}, right = {0};
        memset(scratch, 0, 8*(size_t)nc*sizeof(double));
        read_half(&left, x, size, rs, cs, rows, nl, j, k, target, classification, nc, means[0], scratch);
        read_half(&right, x, size, rs, cs, rows+nl, nr, j, k, target+nl, classification, nc,
                  means[1], scratch+4*nc);
        evaluate(&left, &right, means[0], means[1], nc, p, j, k, out);
        if (directions == 2)
            evaluate(&right, &left, means[1], means[0], nc, p, j, k, out + 2*p);
    }
    return 0;
}
