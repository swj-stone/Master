/* Reuse the frozen one-sketch arithmetic; tally the full-training tie score once. */
#include "document_vote_kernel.c"

API int document_repeated_votes(
    const int32_t *ptr, const int32_t *idx, const uint8_t *labels,
    int n, int p, const int64_t *left, const int64_t *left_offsets,
    const int64_t *right, const int64_t *right_offsets, int repeats, int directions,
    int32_t *peers, double *out, double *pooled
) {
    if (p < 2 || n < 2 || repeats < 1 || (directions != 1 && directions != 2)) return 1;
    Term *terms = calloc((size_t)p, sizeof(Term));
    if (!terms) return 2;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        memset(terms, 0, (size_t)p * sizeof(Term));
        const int64_t *fit = left + left_offsets[repeat];
        const int64_t *held = right + right_offsets[repeat];
        int nf = (int)(left_offsets[repeat + 1] - left_offsets[repeat]);
        int nh = (int)(right_offsets[repeat + 1] - right_offsets[repeat]);
        if (nf < 1 || nh < 1) {
            free(terms);
            return 1;
        }
        int fit_positive = 0, held_positive = 0;
        tally(ptr, idx, labels, fit, nf, 0, terms, &fit_positive);
        tally(ptr, idx, labels, held, nh, 1, terms, &held_positive);
        int32_t *repeat_peers = peers + (size_t)repeat * directions * 2;
        double *repeat_output = out + (size_t)repeat * directions * 2 * p;
        direction(terms, p, ptr, idx, fit, nf, held, nh, 0, fit_positive, held_positive,
                  repeat_peers, repeat_output);
        if (directions == 2) {
            direction(terms, p, ptr, idx, held, nh, fit, nf, 1, held_positive, fit_positive,
                      repeat_peers + 2, repeat_output + 2 * (size_t)p);
        }
    }
    for (int j = 0; j < p; ++j) {
        terms[j].fit_cross = 0;
        terms[j].held_cross = 0;
    }
    int positives = 0;
    for (int row = 0; row < n; ++row) {
        positives += labels[row];
        for (int pos = ptr[row]; pos < ptr[row + 1]; ++pos) {
            ++terms[idx[pos]].fit_cross;
            terms[idx[pos]].held_cross += labels[row];
        }
    }
    double p0 = (double)(n - positives) / n, p1 = (double)positives / n;
    for (int j = 0; j < p; ++j) {
        double frequency = terms[j].fit_cross, ones = terms[j].held_cross;
        double rhs0 = (frequency - ones) - frequency * p0;
        double rhs1 = ones - frequency * p1;
        pooled[j] = (rhs0 * rhs0 + rhs1 * rhs1) / (frequency * (1. - frequency / n) + 1.);
    }
    free(terms);
    return 0;
}
