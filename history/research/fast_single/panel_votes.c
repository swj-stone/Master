/* Sufficient statistics and one-peer conditional gains for a fixed sampled panel. */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static inline double value_at(const char *ptr, int size) {
    if (size == 4) {
        float v;
        memcpy(&v, ptr, 4);
        return v;
    }
    double v;
    memcpy(&v, ptr, 8);
    return v;
}

/* For a dense column use its first sampled value as a numerical origin.
   Work contains sums, squares, then class/response and peer cross-products. */
EXPORT int dense_moments(const char *x, ptrdiff_t row_stride, ptrdiff_t col_stride,
                         int itemsize, const int64_t *rows, int m, int p,
                         const double *target, int nclass, const int64_t *peers, int h,
                         double *work) {
    const int c = nclass ? nclass : 1;
    double *origin = malloc(2 * (size_t)p * sizeof(double));
    if (!origin) return -1;
    double *delta = origin + p;
    const char *first = x + rows[0]*row_stride;
    for (int j=0; j<p; ++j) origin[j] = value_at(first+j*col_stride, itemsize);
    for (int i=0; i<m; ++i) {
        const char *row = x + rows[i]*row_stride;
        double *rhs = work+(2+(nclass ? (int)target[i] : 0))*(size_t)p;
        const double weight = nclass ? 1. : target[i];
        for (int j=0; j<p; ++j) {
            const double d = value_at(row+j*col_stride, itemsize)-origin[j];
            delta[j] = d;
            work[j] += d;
            work[p+j] += d*d;
            rhs[j] += d*weight;
        }
        for (int hidx=0; hidx<h; ++hidx) {
            const double loading = delta[peers[hidx]];
            if (loading == 0.) continue;
            double *cross = work+(2+c+hidx)*(size_t)p;
            for (int j=0; j<p; ++j) cross[j] += loading*delta[j];
        }
    }
    free(origin);
    return 0;
}

static inline int64_t index_at(const void *arr, int size, size_t i) {
    return size==4 ? ((const int32_t *)arr)[i] : ((const int64_t *)arr)[i];
}

/* Iterate nonzeros only. The caller supplies canonical CSR, so every peer
   can appear at most once in a sampled row. */
EXPORT int sparse_moments(const char *data, int size, const void *indices, int isize,
                          const void *indptr, int psize, const int64_t *rows, int m, int p,
                          const double *target, int nclass, const int64_t *peers, int h,
                          double *work) {
    const int c = nclass ? nclass : 1;
    int *panel_slot = calloc((size_t)p, sizeof(int));
    double *values = malloc(h*sizeof(double));
    if (!panel_slot || !values) {free(panel_slot); free(values); return -1;}
    for (int k=0; k<h; ++k) panel_slot[peers[k]] = k+1;
    for (int i=0; i<m; ++i) {
        const int64_t first = index_at(indptr, psize, rows[i]);
        const int64_t last = index_at(indptr, psize, rows[i]+1);
        double *rhs = work+(2+(nclass ? (int)target[i] : 0))*(size_t)p;
        const double weight = nclass ? 1. : target[i];
        memset(values, 0, h*sizeof(double));
        for (int64_t pos=first; pos<last; ++pos) {
            const int j = (int)index_at(indices, isize, pos);
            const double v = value_at(data+pos*size, size);
            work[j] += v;
            work[p+j] += v*v;
            rhs[j] += v*weight;
            if (panel_slot[j]) values[panel_slot[j]-1] = v;
        }
        for (int k=0; k<h; ++k) {
            if (values[k] == 0.) continue;
            double *cross = work+(2+c+k)*(size_t)p;
            for (int64_t pos=first; pos<last; ++pos) {
                int j = (int)index_at(indices, isize, pos);
                cross[j] += values[k]*value_at(data+pos*size, size);
            }
        }
    }
    free(panel_slot);
    free(values);
    return 0;
}

/* Returns 1 for cancellation in unshifted sparse moments: the caller
   recomputes on the same sampled rows with the shifted dense kernel. */
EXPORT int panel_gains(double *work, int p, int m, const double *ysum, int c,
                       const int64_t *peers, int h, int sparse_input, double *tiers) {
    const double penalty = sqrt((double)m);
    const double diagonal = m+penalty;
    double *scale = work+p;
    double *rhs = work+2*(size_t)p;
    for (int j=0; j<p; ++j) {
        const double sq = scale[j], var = sq-work[j]*(work[j]/m);
        if (!isfinite(var)) return -2;
        if (sparse_input && sq>0. && var <= 64.*DBL_EPSILON*sq) return 1;
        scale[j] = var>0. ? sqrt(m/var) : 0.;
        for (int t=0; t<c; ++t) {
            double *b = rhs+t*(size_t)p;
            b[j] = (b[j]-ysum[t]*(work[j]/m))*scale[j];
        }
    }
    for (int j=0; j<p; ++j) {
        if (scale[j]==0.) {tiers[j]=tiers[p+j]=0.;continue;}
        double empty = 0., conditional = 0.;
        for (int t=0; t<c; ++t) {
            double b = rhs[t*(size_t)p+j];
            empty += b*b;
        }
        empty /= diagonal;
        int peers_used = 0;
        for (int a=0; a<h; ++a) {
            const int k = (int)peers[a];
            if (k==j) continue;
            ++peers_used;
            if (scale[k]==0.) {conditional+=empty;continue;}
            const double raw = work[(2+c+a)*(size_t)p+j];
            const double cross = (raw-work[j]*(work[k]/m))*scale[j]*scale[k];
            const double proj = cross/diagonal;
            const double denom = fmax(penalty, diagonal-cross*proj);
            double norm=0.;
            for (int t=0; t<c; ++t) {
                const double *b = rhs+t*(size_t)p;
                const double v = b[j]-proj*b[k];
                norm+=v*v;
            }
            conditional += norm/denom;
        }
        tiers[j] = empty;
        tiers[p+j] = conditional/peers_used;
        if (!isfinite(tiers[j]) || !isfinite(tiers[p+j])) return -2;
    }
    return 0;
}
