/* Compress the columns present in sampled CSR rows before singleton reduction.
   Retain all peer slots (including empty peers) to preserve the original divisor
   and summation order of the conditional stratum. */
#include "../fast_single/panel_votes.c"

EXPORT int compact_votes(const void *data, int size, const void *indices, int isize,
                         const void *indptr, int psize, const int64_t *rows,
                         int m, int p, const double *target, int nclass,
                         const double *sums, const int64_t *peers, int h,
                         double *result) {
    const int c = nclass ? nclass : 1;
    size_t capacity = (size_t)h;
    for (int i=0; i<m && capacity<(size_t)p; ++i)
        capacity += index_at(indptr,psize,rows[i]+1)-index_at(indptr,psize,rows[i]);
    if (capacity>(size_t)p) capacity=p;
    int *slots = calloc((size_t)p, sizeof(int));
    int *original = malloc(capacity*sizeof(int));
    if (!slots || !original) {free(slots);free(original);return -1;}
    int active=0;
    for (int j=0; j<h; ++j) {
        original[active]=peers[j];
        slots[peers[j]]=++active;
    }
    for (int i=0; i<m; ++i) {
        const int64_t a=index_at(indptr,psize,rows[i]), b=index_at(indptr,psize,rows[i]+1);
        for (int64_t u=a; u<b; ++u) {
            const int j=(int)index_at(indices,isize,u);
            if (!slots[j]) {original[active]=j;slots[j]=++active;}
        }
    }
    double *work=calloc((size_t)(2+c+h)*active,sizeof(double));
    double *tiers=malloc(2*(size_t)active*sizeof(double));
    int64_t *local=malloc((size_t)h*sizeof(int64_t));
    double *values=malloc((size_t)h*sizeof(double));
    int status=-1;
    if (!work || !tiers || !local || !values) goto finish;
    for (int a=0;a<h;++a) local[a]=a;
    for (int i=0;i<m;++i) {
        const int64_t a=index_at(indptr,psize,rows[i]), b=index_at(indptr,psize,rows[i]+1);
        const int t=nclass ? (int)target[i] : 0;
        const double weight=nclass ? 1. : target[i];
        double *rhs=work+(2+t)*(size_t)active;
        memset(values,0,(size_t)h*sizeof(double));
        for (int64_t u=a;u<b;++u) {
            const int j=slots[index_at(indices,isize,u)]-1;
            const double v=value_at((const char *)data+u*size,size);
            work[j]+=v;
            work[active+j]+=v*v;
            rhs[j]+=v*weight;
            if (j<h) values[j]=v;
        }
        for (int k=0;k<h;++k) {
            if (values[k]==0.) continue;
            double *cross=work+(2+c+k)*(size_t)active;
            for (int64_t u=a;u<b;++u) {
                const int j=slots[index_at(indices,isize,u)]-1;
                cross[j]+=values[k]*value_at((const char *)data+u*size,size);
            }
        }
    }
    status=panel_gains(work,active,m,sums,c,local,h,1,tiers);
    if (!status)
        for(int j=0;j<active;++j) {
            result[original[j]]=tiers[j];
            result[p+original[j]]=tiers[active+j];
        }
finish:
    free(values);free(local);free(tiers);free(work);free(original);free(slots);
    return status;
}
