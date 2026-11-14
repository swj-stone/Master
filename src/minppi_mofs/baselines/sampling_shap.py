"""Global feature ranking from SHAP's IME SamplingExplainer (Štrumbelj & Kononenko).

Explain a ranking-only sample with a fixed trained predictor and empirical training
background. Absolute local attributions are averaged over observations and outputs.
"""
import numpy as np

from minppi_mofs.baselines._selection_context import Selection
from minppi_mofs.baselines.pfi import sampled_rows


BACKGROUND_LIMIT = 128
EXPLANATION_LIMIT = 32
PAIRS_PER_FEATURE = 12


def select(context):
    """Score original columns using paired, random background/permutation predictions."""
    import shap

    columns = list(context.columns)
    if (not columns or not context.train.columns.is_unique or list(context.train.columns) != columns
            or list(context.ranking.columns) != columns or not len(context.train) or not len(context.ranking)):
        raise ValueError('SamplingSHAP requires aligned, nonempty training and ranking features')
    regression = context.task == 'regression'
    background_seed, ranking_seed, shap_seed = np.random.SeedSequence(context.seed).spawn(3)
    background_rows = np.sort(np.random.default_rng(background_seed).choice(
        len(context.train), min(BACKGROUND_LIMIT, len(context.train)), replace=False))
    ranking_rows = sampled_rows(context.yr, EXPLANATION_LIMIT, regression, ranking_seed)
    background = np.ascontiguousarray(context.train.iloc[background_rows], dtype=np.float32)
    explain = np.ascontiguousarray(context.ranking.iloc[ranking_rows], dtype=np.float32)
    predict = context.full_model.predict if regression else context.full_model.predict_proba

    initialising = True
    counts = {stage: {'rows': 0, 'calls': 0} for stage in ('background', 'explained', 'sampling')}

    def counted_predict(matrix):
        # SHAP evaluates the background at construction. Per explained point it
        # gets f(x) once, then predicts paired synthetic examples (at least two).
        stage = 'background' if initialising else 'explained' if len(matrix) == 1 else 'sampling'
        counts[stage]['rows'] += len(matrix)
        counts[stage]['calls'] += 1
        return np.asarray(predict(np.ascontiguousarray(matrix, dtype=np.float32)), dtype=np.float64)

    budget = PAIRS_PER_FEATURE * len(columns)
    scores = np.zeros(len(columns), dtype=float)
    # SamplingExplainer 0.52 uses the global NumPy RNG for permutations and
    # background-row draws. Routes run in separate processes; restore caller state.
    previous_state = np.random.get_state()
    random_seed = int(shap_seed.generate_state(1)[0])
    try:
        np.random.seed(random_seed)
        explainer = shap.SamplingExplainer(counted_predict, background)
        initialising = False
        for i, point in enumerate(explain):
            phi = np.asarray(explainer.shap_values(point[None, :], nsamples=budget, silent=True))
            if phi.shape[:2] != (1, len(columns)) or not np.isfinite(phi).all():
                raise ValueError(f'SamplingSHAP returned invalid attributions for ranking row {ranking_rows[i]}')
            # A scalar predictor has one output; predict_proba returns all classes.
            values = phi.reshape(len(columns), -1)
            scores += np.abs(values).mean(axis=1)
            context.progress('sampling_shap', covered=i+1, total=len(explain),
                             sampling_prediction_rows=counts['sampling']['rows'])
    finally:
        np.random.set_state(previous_state)

    scores /= len(explain)
    return Selection(scores=scores, diagnostics=dict(
        implementation='shap.SamplingExplainer', shap_version=shap.__version__,
        aggregation='mean absolute local attribution, equal weight per explained row and model output',
        model_output='prediction' if regression else 'predict_proba',
        seed=context.seed, shap_seed=random_seed, nsamples_per_explanation=budget,
        pairs_per_input_feature_budget=PAIRS_PER_FEATURE,
        ranking_rows=ranking_rows.tolist(), background_rows=background_rows.tolist(),
        paired_samples=counts['sampling']['rows']//2,
        sampling_prediction_rows=counts['sampling']['rows'], sampling_prediction_calls=counts['sampling']['calls'],
        background_prediction_rows=counts['background']['rows'], explained_prediction_rows=counts['explained']['rows'],
        prediction_rows=sum(v['rows'] for v in counts.values()), prediction_calls=sum(v['calls'] for v in counts.values())))
