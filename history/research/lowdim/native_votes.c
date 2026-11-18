/* Sufficient statistics for the zero-peer and mean-singleton ridge votes. */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#define MAX_P 32
#define MAX_ROWS 768

static uint64_t next_word(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static size_t uniform_offset(uint64_t *state, size_t span) {
    uint64_t acceptance = (-(uint64_t)span) % span;
    uint64_t draw;
    do { draw = next_word(state); } while (draw < acceptance);
    return draw % span;
}

static double value_at(const char *ptr, int size) {
    if (size == 4) {
        float val;
        memcpy(&val, ptr, sizeof(val));
        return val;
    }
    double val;
    memcpy(&val, ptr, sizeof(val));
    return val;
}

/* Numeric y is float64; classification labels are consecutive integer IDs.
   Each feature row is read just once during moment construction. */
EXPORT int lowdim_votes(const char *X, ptrdiff_t row_step, ptrdiff_t col_step,
                        int item_size, const char *y, ptrdiff_t y_step,
                        size_t n, int p, int num_classes, uint64_t seed,
                        double *scores, double *layer0, double *layer1, int64_t *columns) {
    if (p < 1 || p > MAX_P || !n || (item_size != 4 && item_size != 8) || num_classes < 0)
        return -1;
    const size_t m = n < MAX_ROWS ? n : MAX_ROWS;
    size_t rows[MAX_ROWS];
    double target[MAX_ROWS], target_sum = 0., target_sq = 0., amplitude = 0.;
    /* Only classes present in the sample need columns in X'Y. */
    double labels[MAX_ROWS];
    size_t class_id[MAX_ROWS], counts[MAX_ROWS] = {0}, classes = 0;
    for (size_t i = 0; i < m; i++) {
        size_t left = i * n / m, right = (i+1) * n / m;
        rows[i] = left + (right-left > 1 ? uniform_offset(&seed, right-left) : 0);
        target[i] = value_at(y+(ptrdiff_t)rows[i]*y_step, 8);
        if (!isfinite(target[i])) return -2;
        if (num_classes) {
            size_t c = 0;
            while (c < classes && labels[c] != target[i]) c++;
            if (c == classes) labels[classes++] = target[i];
            counts[c]++;
            class_id[i] = c;
        } else {
            amplitude = fmax(amplitude, fabs(target[i]));
        }
    }
    if (!num_classes) classes = 1;
    double rhs[MAX_ROWS][MAX_P];
    memset(rhs, 0, classes * sizeof(rhs[0]));
    double gram[MAX_P][MAX_P] = {{0}}, sums[MAX_P] = {0}, origin[MAX_P];
    const char *first = X+(ptrdiff_t)rows[0]*row_step;
    for (int j = 0; j < p; j++) origin[j] = value_at(first+j*col_step, item_size);
    for (size_t i = 0; i < m; i++) {
        const char *row = X + (ptrdiff_t)rows[i]*row_step;
        double delta[MAX_P];
        for (int j = 0; j < p; j++) {
            delta[j] = value_at(row+j*col_step, item_size) - origin[j];
            if (!isfinite(delta[j])) return -2;
            sums[j] += delta[j];
        }
        for (int j = 0; j < p; j++)
            for (int k = 0; k <= j; k++) gram[j][k] += delta[j]*delta[k];
        if (num_classes) {
            for (int j = 0; j < p; j++) rhs[class_id[i]][j] += delta[j];
        } else {
            const double t = amplitude ? target[i]/amplitude - target[0]/amplitude : 0.;
            target_sum += t;
            target_sq += t*t;
            for (int j = 0; j < p; j++) rhs[0][j] += delta[j]*t;
        }
    }
    double scale[MAX_P], penalized[MAX_P], reg = sqrt((double)m);
    double yss = fmax(0., target_sq - target_sum*target_sum/m);
    double y_scale = !num_classes && yss > 0. ? sqrt(m/yss) : 1.;
    for (int j = 0; j < p; j++) {
        for (int k = 0; k <= j; k++) gram[j][k] -= sums[j]*sums[k]/m;
        if (!isfinite(gram[j][j])) return -3;
        scale[j] = gram[j][j] > 0 ? sqrt(m/gram[j][j]) : 0.;
        penalized[j] = (scale[j] > 0. ? (double)m : 0.) + reg;
        for (size_t c = 0; c < classes; c++) {
            double center = num_classes ? (double)counts[c] : target_sum;
            rhs[c][j] = ((rhs[c][j] - center*sums[j]/m)*scale[j])*y_scale;
        }
    }
    for (int j = 0; j < p; j++) {
        layer0[j] = 0.;
        for (size_t c = 0; c < classes; c++) layer0[j] += rhs[c][j]*rhs[c][j];
        layer0[j] /= penalized[j];
        double total = 0.;
        for (int k = 0; k < p; k++) {
            if (k == j) continue;
            const double cross = (j>k ? gram[j][k] : gram[k][j])*scale[j]*scale[k];
            const double proj = cross / penalized[k];
            const double denom = fmax(reg, penalized[j] - cross*proj);
            double squared = 0.;
            for (size_t c = 0; c < classes; c++) {
                double resid = rhs[c][j] - proj*rhs[c][k];
                squared += resid*resid;
            }
            total += squared/denom;
        }
        layer1[j] = p>1 ? total/(p-1) : layer0[j];
        scores[j] = fmin(layer0[j], layer1[j]);
        if (!isfinite(scores[j])) return -3;
    }
    int ranking[MAX_P];
    for (int j = 0; j < p; j++) {
        int k = j;
        while (k>0 && scores[ranking[k-1]] < scores[j]) {
            ranking[k] = ranking[k-1];
            k--;
        }
        ranking[k] = j;
    }
    int width = p;
    if (scores[ranking[0]] > 0.) {
        double sum = 0., squares = 0.;
        for (int j = 0; j < p; j++) {
            double w = scores[j]/scores[ranking[0]];
            sum += w;
            squares += w*w;
        }
        width = (int)ceil(sum*sum/squares);
        if (width>p) width = p;
    }
    int selected[MAX_P] = {0};
    for (int j=0; j<width; j++) selected[ranking[j]]=1;
    int pos=0;
    for (int j=0; j<p; j++) if (selected[j]) columns[pos++]=j;
    return width;
}
