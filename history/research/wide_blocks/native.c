/* Ridge selection on centered, standardized cross-products in caller-owned buffers. */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#define MAX_COLUMNS 256

/* Match the existing low-dimensional selector's stratified row stream. */
EXPORT void wide_rows(size_t n, size_t m, uint64_t state, int64_t *rows) {
    for (size_t i = 0; i < m; i++) {
        size_t lo = i*n/m, span = (i+1)*n/m-lo;
        uint64_t word = 0;
        if (span > 1) {
            do {
                uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
                z = (z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
                z = (z^(z>>27))*UINT64_C(0x94d049bb133111eb);
                word = z^(z>>31);
            } while (word < (-(uint64_t)span)%span);
        }
        rows[i] = (int64_t)(lo + word%span);
    }
}

static int better(double a, double b) {
    return a-b > 32.*DBL_EPSILON*fmax(fabs(a),fabs(b));
}
static double norm2(const double *b, int c) {
    double s = 0.;
    for (int t=0; t<c; ++t) s += b[t]*b[t];
    return s;
}
static double conditional(const double *a, const double *b, int p, int c,
                          int j, int k, double reg) {
    double cross = a[j*p+k], factor = cross/a[k*p+k], s = 0.;
    for (int t=0; t<c; ++t) {
        double x = b[j*c+t]-factor*b[k*c+t];
        s += x*x;
    }
    return s/fmax(reg,a[j*p+j]-factor*cross);
}
static void eliminate(double *a, double *b, int p, int c, int k,
                      const int *active, int n, double reg) {
    for (int u=0; u<n; ++u) {
        int j=active[u];
        double factor=a[j*p+k]/a[k*p+k];
        for (int v=0; v<=u; ++v) {
            int l=active[v];
            a[j*p+l] -= factor*a[k*p+l];
            a[l*p+j] = a[j*p+l];
        }
        a[j*p+j] = fmax(reg,a[j*p+j]);
        for (int t=0; t<c; ++t) b[j*c+t] -= factor*b[k*c+t];
    }
}
static double criterion(double gain, double loss, size_t m, double cost) {
    return -(double)m*log1p(-fmin(1.-DBL_EPSILON, fmax(0.,gain/loss))) - cost;
}

/* a=Z'Z+sqrt(m)I, b=Z'Y. The first two vote rows are empty and singleton.
   Return zero for the no-signal fallback; caller then retains the original input. */
EXPORT int wide_path(double *a, double *b, int p, int c, size_t m,
                     double loss, int fill, double *votes, int64_t *order,
                     uint8_t *sizes, int64_t *cap_out) {
    if (p<1 || p>MAX_COLUMNS || c<1 || c>768 || !m) return -1;
    double reg=sqrt((double)m), *empty=votes, *single=votes+p, *score=votes+2*p;
    double gains[MAX_COLUMNS], max_score=0., best_seed=-1.;
    int first=0, second=0, active[MAX_COLUMNS];
    for (int j=0; j<p; ++j) {
        empty[j]=norm2(b+j*c,c)/a[j*p+j]; active[j]=j;
    }
    for (int j=0; j<p; ++j) {
        double total=0.;
        for (int k=0; k<p; ++k) {
            if (j==k) continue;
            double g=conditional(a,b,p,c,j,k,reg);
            total += g;
            if (j<k && better(empty[k]+g,best_seed)) {
                best_seed=empty[k]+g; first=j; second=k;
            }
        }
        single[j]=p>1 ? total/(p-1) : empty[j];
        score[j]=fmin(empty[j],single[j]);
        if (!isfinite(empty[j]) || !isfinite(single[j])) return -1;
        max_score=fmax(max_score,score[j]);
    }
    if (max_score<=0. || m<2) { *cap_out=p; return 0; }
    double s=0., ss=0.;
    for (int j=0; j<p; ++j) {
        double w=score[j]/max_score; s+=w; ss+=w*w;
    }
    int cap=(int)fmin(p,ceil(s*s/ss)); *cap_out=cap;
    if (cap==1) {
        first=0;
        for (int j=1; j<p; ++j) if (better(empty[j],empty[first])) first=j;
        order[0]=first; sizes[0]=1;
        return 1;
    }
    order[0]=first; order[1]=second; sizes[0]=2;
    loss=fmax(0.,loss-best_seed);
    int width=2, conditioned=0, count=p;
    const double penalty=log((double)m);
    while (width<cap) {
        while (conditioned<width) {
            int k=(int)order[conditioned++], kept=0;
            for (int u=0; u<count; ++u)
                if (active[u]!=k) active[kept++]=active[u];
            count=kept;
            eliminate(a,b,p,c,k,active,count,reg);
        }
        if (!(loss>0.)) {
            if (fill) for (int u=0; width<cap; ++u) {
                sizes[width]=1; order[width++]=active[u];
            }
            break;
        }
        first=active[0]; second=-1;
        for (int u=0; u<count; ++u) {
            int j=active[u];
            gains[j]=norm2(b+j*c,c)/a[j*p+j];
            if (better(gains[j],gains[first])) first=j;
        }
        double gain=gains[first], best=criterion(gain,loss,m,penalty);
        if (cap-width>=2) {
            int pair1=-1, pair2=-1;
            double pair_gain=-1.;
            for (int u=0; u<count; ++u) for (int v=u+1; v<count; ++v) {
                int j=active[u], k=active[v];
                double joint=gains[j]+conditional(a,b,p,c,k,j,reg);
                if (better(joint,pair_gain)) { pair_gain=joint; pair1=j; pair2=k; }
            }
            double score2=criterion(pair_gain,loss,m,2.*penalty);
            if (better(score2,best)) { first=pair1; second=pair2; best=score2; gain=pair_gain; }
        }
        if (!fill && !(best>0.)) break;
        sizes[width]=second<0 ? 1 : 2;
        order[width++]=first;
        if (second>=0) order[width++]=second;
        loss=fmax(0.,loss-gain);
    }
    return width;
}
