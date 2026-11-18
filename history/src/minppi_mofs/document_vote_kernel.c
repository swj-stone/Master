/* Binary-presence minPI using canonical CSR indices and independent one-hot coordinates. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct {
    double df[2], full[2], root, mean, single[2], held_rhs[2], held_square;
    int fit_cross, held_cross;
} Term;

static void tally(const int32_t *ptr, const int32_t *idx, const uint8_t *labels,
                  const int64_t *rows, int n, int side, Term *t, int *n_class1) {
    for (int i=0; i<n; ++i) {
        int row = (int)rows[i], label = labels[row];
        *n_class1 += label;
        for (int j=ptr[row]; j<ptr[row+1]; ++j) {
            t[idx[j]].df[side] += 1.;
            t[idx[j]].full[side] += label;
        }
    }
}

static void cooccur(const int32_t *ptr, const int32_t *idx, const int64_t *rows, int n,
                    int anchor, int on_fit, Term *t) {
    for (int i=0; i<n; ++i) {
        int row = (int)rows[i];
        int l=ptr[row], r=ptr[row+1], end=r;
        while (l<r) {
            int mid=l+(r-l)/2;
            if (idx[mid] < anchor) l=mid+1;
            else r=mid;
        }
        if (l<end && idx[l]==anchor) {
            for (int j=ptr[row]; j<end; ++j) {
                if (on_fit) ++t[idx[j]].fit_cross;
                else ++t[idx[j]].held_cross;
            }
        }
    }
}

static void direction(Term *t, int p, const int32_t *ptr, const int32_t *idx,
                      const int64_t *fit, int nf, const int64_t *held, int nh,
                      int f, int ncfit, int ncheld, int32_t *peers, double *out) {
    int h=1-f, anchor=0, alternate=1;
    double best=-1., second=-1.;
    double prior[2] = {(double)(nf-ncfit)/nf, (double)ncfit/nf};
    double held_prior[2] = {nh-ncheld, ncheld};
    for (int j=0; j<p; ++j) {
        Term *s=t+j;
        s->fit_cross=s->held_cross=0;
        s->mean=s->df[f]/nf;
        s->root=sqrt(s->df[f]*(1.-s->mean)+1.);
        s->held_square=(s->df[h]*(1.-2.*s->mean)+nh*(s->mean*s->mean))/(s->root*s->root);
        double f_joint[2]={s->df[f]-s->full[f], s->full[f]};
        double h_joint[2]={s->df[h]-s->full[h], s->full[h]};
        for (int c=0; c<2; ++c) {
            s->single[c]=(f_joint[c]-s->df[f]*prior[c])/s->root;
            s->held_rhs[c]=(h_joint[c]-s->df[h]*prior[c]
                       - s->mean*(held_prior[c]-nh*prior[c]))/s->root;
        }
        double strength=s->single[0]*s->single[0]+s->single[1]*s->single[1];
        if (strength > best) {
            second=best; alternate=anchor; best=strength; anchor=j;
        } else if (strength > second) {
            second=strength; alternate=j;
        }
    }
    peers[0]=anchor; peers[1]=alternate;
    cooccur(ptr,idx,fit,nf,anchor,1,t);
    cooccur(ptr,idx,held,nh,anchor,0,t);
    t[anchor].fit_cross=t[alternate].fit_cross;
    t[anchor].held_cross=t[alternate].held_cross;
    for (int j=0; j<p; ++j) {
        Term *s=t+j, *a=t+(j==anchor?alternate:anchor);
        double r=(s->fit_cross-s->mean*a->df[f])/s->root/a->root;
        double cross=(s->held_cross-s->mean*a->df[h]-a->mean*s->df[h]+nh*s->mean*a->mean)/s->root/a->root;
        double norm=s->held_square-2.*r*cross+(r*r)*a->held_square;
        double zs=0., zsq=0., os=0., osq=0.;
        for (int c=0; c<2; ++c) {
            double b=(s->single[c]-r*a->single[c])/(1.-r*r);
            double rhs=s->held_rhs[c]-r*a->held_rhs[c]-(cross-r*a->held_square)*a->single[c];
            zs+=s->held_rhs[c]*s->single[c]; zsq+=s->single[c]*s->single[c];
            os+=rhs*b; osq+=b*b;
        }
        out[j]=(2.*zs-s->held_square*zsq)/nh;
        out[p+j]=(2.*os-norm*osq)/nh;
    }
}

API int document_votes(const int32_t *ptr, const int32_t *idx, const uint8_t *labels,
                       int n, int p, const int64_t *left, int nl, const int64_t *right,
                       int nr, int directions, int32_t *peers, double *out, double *pooled) {
    if (p<2 || n<2 || nl<1 || nr<1 || (directions!=1 && directions!=2)) return 1;
    Term *t=calloc((size_t)p,sizeof(Term));
    if (!t) return 2;
    int nleft1=0, nright1=0;
    tally(ptr,idx,labels,left,nl,0,t,&nleft1);
    tally(ptr,idx,labels,right,nr,1,t,&nright1);
    direction(t,p,ptr,idx,left,nl,right,nr,0,nleft1,nright1,peers,out);
    if (directions==2)
        direction(t,p,ptr,idx,right,nr,left,nl,1,nright1,nleft1,peers+2,out+2*(size_t)p);
    /* Reuse scratch arrays for the full training tie breaker. */
    for (int j=0; j<p; ++j) {t[j].fit_cross=0; t[j].held_cross=0;}
    int positives=0;
    for (int i=0; i<n; ++i) {
        positives+=labels[i];
        for (int k=ptr[i]; k<ptr[i+1]; ++k) {
            ++t[idx[k]].fit_cross;
            t[idx[k]].held_cross+=labels[i];
        }
    }
    double p0=(double)(n-positives)/n, p1=(double)positives/n;
    for (int j=0; j<p; ++j) {
        double df=t[j].fit_cross, ones=t[j].held_cross;
        double b0=(df-ones)-df*p0, b1=ones-df*p1;
        pooled[j]=(b0*b0+b1*b1)/(df*(1.-df/n)+1.);
    }
    free(t);
    return 0;
}
