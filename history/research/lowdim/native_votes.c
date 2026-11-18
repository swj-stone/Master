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

/* Candidates are visited in original column order. Treat differences within
   the rounding error of a short ridge calculation as ties. */
static int higher_gain(double value, double best) {
    return value - best > 32.*DBL_EPSILON*fmax(fabs(value), fabs(best));
}

/* Remove one selected pivot and update the conditional Gram matrix, residual
   responses and singleton gains for all columns still available. */
static int condition_on(int pivot, int *remaining, int count, size_t classes, double reg,
                        double gram[][MAX_P], double rhs[][MAX_P], double *gains) {
    int kept = 0;
    for (int a = 0; a < count; a++) {
        if (remaining[a] != pivot) remaining[kept++] = remaining[a];
    }
    for (int a = 0; a < kept; a++) {
        const int j = remaining[a];
        const double factor = gram[j][pivot]/gram[pivot][pivot];
        for (int b = 0; b <= a; b++) {
            const int k = remaining[b];
            gram[j][k] -= factor*gram[pivot][k];
            gram[k][j] = gram[j][k];
        }
        gram[j][j] = fmax(reg, gram[j][j]);
        double squared = 0.;
        for (size_t c = 0; c < classes; c++) {
            rhs[c][j] -= factor*rhs[c][pivot];
            squared += rhs[c][j]*rhs[c][j];
        }
        gains[j] = squared/gram[j][j];
    }
    return kept;
}

static double block_score(double m, double gain, double loss, double penalty) {
    const double fraction = fmin(1.-DBL_EPSILON, fmax(0., gain/loss));
    return -m*log1p(-fraction) - penalty;
}

/* The maximum gain within each block size also maximizes its log score.
   Return its column(s) in ascending index order followed by their joint gain. */
static double best_conditional_pair(const int *remaining, int count, size_t classes, double reg,
                                    const double gram[][MAX_P], const double rhs[][MAX_P],
                                    const double *gains, int *first, int *second) {
    double best = -1.;
    for (int a = 0; a < count; a++) {
        const int j = remaining[a];
        for (int b = a+1; b < count; b++) {
            const int k = remaining[b];
            const double cross = gram[k][j];
            const double factor = cross/gram[j][j];
            const double denom = fmax(reg, gram[k][k] - cross*factor);
            double squared = 0.;
            for (size_t c = 0; c < classes; c++) {
                const double delta = rhs[c][k] - factor*rhs[c][j];
                squared += delta*delta;
            }
            const double gain = gains[j] + squared/denom;
            if (higher_gain(gain, best)) {
                best = gain;
                *first = j;
                *second = k;
            }
        }
    }
    return best;
}

/* order[0:2] contains the exact best pair. Sequential Schur complements
   condition on its columns and on any blocks added later, without refitting. */
static int extend_path(int p, int cap, size_t classes, double reg,
                       double gram[][MAX_P], double rhs[][MAX_P], const double *scale,
                       const double *penalized, int *order, int blocks, double m,
                       double loss, uint8_t *block_sizes) {
    int remaining[MAX_P], count = p;
    double gains[MAX_P];
    for (int j = 0; j < p; j++) {
        remaining[j] = j;
        gram[j][j] = penalized[j];
        for (int k = 0; k < j; k++) {
            gram[j][k] = gram[j][k]*scale[j]*scale[k];
            gram[k][j] = gram[j][k];
        }
    }
    const double penalty = blocks ? log(m) : 0.;
    int width = 2, conditioned = 0;
    while (width < cap) {
        while (conditioned < width) {
            count = condition_on(order[conditioned++], remaining, count, classes, reg,
                                 gram, rhs, gains);
        }
        int first = remaining[0], second = -1;
        double gain = gains[first];
        for (int a = 1; a < count; a++) {
            int j = remaining[a];
            if (higher_gain(gains[j], gain)) {
                gain = gains[j];
                first = j;
            }
        }
        if (blocks) {
            if (!(loss > 0.)) break;
            double score = block_score(m, gain, loss, penalty);
            if (cap-width >= 2) {
                int j = -1, k = -1;
                const double pair_gain = best_conditional_pair(remaining, count, classes, reg,
                                                              gram, rhs, gains, &j, &k);
                const double pair_score = block_score(m, pair_gain, loss, 2.*penalty);
                if (higher_gain(pair_score, score)) {
                    score = pair_score;
                    first = j;
                    second = k;
                    gain = pair_gain;
                }
            }
            if (!(score > 0.)) break;
            loss = fmax(0., loss-gain);
            if (block_sizes) block_sizes[width] = second >= 0 ? 2 : 1;
        }
        order[width++] = first;
        if (second >= 0) order[width++] = second;
    }
    return width;
}

/* Numeric y is float64; classification labels are consecutive integer IDs.
   Each feature row is read just once during moment construction. */
static int lowdim_support(const char *X, ptrdiff_t row_step, ptrdiff_t col_step,
                        int item_size, const char *y, ptrdiff_t y_step,
                        size_t n, int p, int num_classes, uint64_t seed,
                        double *scores, double *layer0, double *layer1, int64_t *columns,
                        int path_mode, int64_t *selection_order, uint8_t *block_sizes) {
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
    }
    double best_pair = -1.;
    int pair_first = 0, pair_second = 0;
    for (int j = 0; j < p; j++) {
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
            const double conditional = squared/denom;
            total += conditional;
            if (path_mode && j < k) {
                const double joint = layer0[k] + conditional;
                if (!isfinite(joint)) return -3;
                if (higher_gain(joint, best_pair)) {
                    best_pair = joint;
                    pair_first = j;
                    pair_second = k;
                }
            }
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
    const int use_blocks = path_mode == 2;
    /* The block ABI reserves one extra column slot for the original cap. */
    if (use_blocks) columns[p] = width;
    const int block_path = use_blocks && m >= 2 && scores[ranking[0]] > 0.;
    if ((path_mode == 1 && width < p) || block_path) {
        if (width == 1) {
            double best = -1.;
            for (int j = 0; j < p; j++) {
                if (higher_gain(layer0[j], best)) {
                    best = layer0[j];
                    ranking[0] = j;
                }
            }
            if (block_sizes) block_sizes[0] = 1;
        } else {
            ranking[0] = pair_first;
            ranking[1] = pair_second;
            if (block_sizes) block_sizes[0] = 2;
            if (width > 2) {
                double loss = 0.;
                if (use_blocks) {
                    /* A standardized nonconstant target has energy m. For
                       centered one-hot columns, sum their individual SS. */
                    loss = (double)m;
                    if (num_classes) {
                        for (size_t c = 0; c < classes; c++)
                            loss -= (double)counts[c]*(double)counts[c]/m;
                    } else if (!(yss > 0.)) {
                        loss = 0.;
                    }
                    loss = fmax(0., loss-best_pair);
                }
                width = extend_path(p, width, classes, reg, gram, rhs, scale, penalized,
                                    ranking, use_blocks, (double)m, loss, block_sizes);
            }
        }
        if (selection_order) {
            for (int j = 0; j < width; j++) selection_order[j] = ranking[j];
        }
    }
    int selected[MAX_P] = {0};
    for (int j=0; j<width; j++) selected[ranking[j]]=1;
    int pos=0;
    for (int j=0; j<p; j++) if (selected[j]) columns[pos++]=j;
    return width;
}

EXPORT int lowdim_votes(const char *X, ptrdiff_t row_step, ptrdiff_t col_step,
                        int item_size, const char *y, ptrdiff_t y_step,
                        size_t n, int p, int num_classes, uint64_t seed,
                        double *scores, double *layer0, double *layer1, int64_t *columns) {
    return lowdim_support(X, row_step, col_step, item_size, y, y_step, n, p, num_classes,
                          seed, scores, layer0, layer1, columns, 0, NULL, NULL);
}

EXPORT int lowdim_pairs(const char *X, ptrdiff_t row_step, ptrdiff_t col_step,
                        int item_size, const char *y, ptrdiff_t y_step,
                        size_t n, int p, int num_classes, uint64_t seed,
                        double *scores, double *layer0, double *layer1, int64_t *columns,
                        int64_t *selection_order) {
    return lowdim_support(X, row_step, col_step, item_size, y, y_step, n, p, num_classes,
                          seed, scores, layer0, layer1, columns, 1, selection_order, NULL);
}

EXPORT int lowdim_blocks(const char *X, ptrdiff_t row_step, ptrdiff_t col_step,
                         int item_size, const char *y, ptrdiff_t y_step,
                         size_t n, int p, int num_classes, uint64_t seed,
                         double *scores, double *layer0, double *layer1, int64_t *columns,
                         int64_t *selection_order, uint8_t *block_sizes) {
    return lowdim_support(X, row_step, col_step, item_size, y, y_step, n, p, num_classes,
                          seed, scores, layer0, layer1, columns, 2, selection_order, block_sizes);
}
