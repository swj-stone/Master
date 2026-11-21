/* Exact singleton ridge gains for a tile of centered, standardized cross-products. */
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void sample_rows(int64_t n, int m, uint64_t seed, int64_t *rows) {
    for (int i=0; i<m; ++i) {
        int64_t left = i*n/m;
        uint64_t span = (i+1)*n/m - left;
        uint64_t offset = 0;
        if (span > 1) {
            uint64_t draw, acceptance = (-span) % span;
            do {
                uint64_t z = (seed += UINT64_C(0x9e3779b97f4a7c15));
                z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
                draw = z ^ (z >> 31);
            } while (draw < acceptance);
            offset = draw % span;
        }
        rows[i] = left + offset;
    }
}

EXPORT void sum_tile(const double *gram, int a, int b, int na, int nb,
                     const double *rhs, int classes, const int64_t *weights,
                     const double *center, double penalty, double diagonal,
                     double *totals) {
    const double inv_diag = 1./diagonal;
    for (int i=0; i<na; ++i) {
        int j=a+i;
        double total=0.;
        int first = a == b ? i+1 : 0;
        for (int l=first; l<nb; ++l) {
            int k=b+l;
            double cross=gram[i*nb+l] - center[j]*center[k];
            double proj=cross*inv_diag;
            double inv_residual=1./fmax(penalty, diagonal-cross*proj);
            double gain_j=0., gain_k=0.;
            for (int c=0; c<classes; ++c) {
                double rj=rhs[j*classes+c], rk=rhs[k*classes+c];
                double dj=rj-proj*rk, dk=rk-proj*rj;
                gain_j += dj*dj;
                gain_k += dk*dk;
            }
            total += weights[k]*gain_j*inv_residual;
            totals[k] += weights[j]*gain_k*inv_residual;
        }
        totals[j] += total;
    }
}
