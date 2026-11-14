"""Exact frozen-model coalition predictions for the permutation feature selectors."""
import json

import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score


def _project_model(model, active, width):
    """Renumber numeric tree splits without changing any branch, leaf, or tree order."""
    original = model.get_booster()
    if not active or len(active) == width:
        return model
    payload = json.loads(original.save_raw(raw_format="json"))
    learner = payload["learner"]
    booster = learner["gradient_booster"]
    if booster["name"] != "gbtree":
        return model
    forest = booster["model"]
    if any(forest.get("cats", {}).values()) or "c" in learner["feature_types"]:
        # Category dictionaries also reference feature coordinates.
        return model
    remap = {feature: pos for pos, feature in enumerate(active)}
    for tree in forest["trees"]:
        for node, child in enumerate(tree["left_children"]):
            if child >= 0:
                old = tree["split_indices"][node]
                tree["split_indices"][node] = remap[old]
        tree["tree_param"]["num_feature"] = str(len(active))
    learner["learner_model_param"]["num_feature"] = str(len(active))
    for key in ("feature_names", "feature_types"):
        if learner[key]:
            learner[key] = [learner[key][i] for i in active]
    projected = clone(model)
    projected.load_model(bytearray(json.dumps(payload, separators=(",", ":")).encode()))
    # Model serialization does not preserve runtime device and thread settings.
    params = json.loads(original.save_config())["learner"]["generic_param"]
    projected.get_booster().set_param({"device": params["device"], "nthread": params["nthread"]})
    return projected


def weighted_auc(labels, pred, weights):
    """Macro OvR (binary positive-class AUC) on the benchmark's full class sample.

    Preserve the reference ROC construction, including stable tie handling,
    intermediate-point pruning and the trapezoidal reduction order. One sort
    per column handles the multiclass case without repeating sklearn validation.
    """
    nonzero = weights != 0
    labels, weights = labels[nonzero], weights[nonzero]
    scores = pred[nonzero]
    if scores.shape[1] == 2:
        scores = scores[:, 1:]
        classes = np.array([1])
    else:
        classes = np.arange(scores.shape[1])
    order = np.argsort(-scores, axis=0, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    positives = labels[order] == classes
    sorted_weights = weights[order]
    tp = np.cumsum(positives * sorted_weights, axis=0)
    fp = np.cumsum((~positives) * sorted_weights, axis=0)
    ends = np.vstack((sorted_scores[:-1] != sorted_scores[1:], np.ones(len(classes), dtype=bool)))
    auc = np.empty(len(classes))
    for c in range(len(classes)):
        tps = tp[ends[:, c], c]
        fps = fp[ends[:, c], c]
        if len(tps) > 2:
            corners = np.r_[True, (np.diff(tps, 2) != 0) | (np.diff(fps, 2) != 0), True]
            tps, fps = tps[corners], fps[corners]
        tpr, fpr = np.r_[0., tps] / tps[-1], np.r_[0., fps] / fps[-1]
        auc[c] = np.trapezoid(tpr, fpr)
    return float(np.mean(auc))


class FrozenMaskPredictor:
    """Pack only features read by the booster and batch its masked predictions."""

    def __init__(self, model, x, fill, active, *, regression):
        active = tuple(active)
        if x.ndim != 2 or len(fill) != x.shape[1] or x.shape[1] != model.get_booster().num_features():
            raise ValueError("Ranking inputs and fitted model schema differ")
        self.model = _project_model(model, active, x.shape[1])
        self.regression = regression
        if self.model is model:
            self.x, self.fill = x, fill
            self.positions = np.arange(x.shape[1])
        else:
            self.x = np.ascontiguousarray(x[:, active])
            self.fill = fill[list(active)]
            self.positions = np.full(x.shape[1], -1, dtype=np.intp)
            self.positions[list(active)] = np.arange(len(active))
        self.physical_width = self.x.shape[1]
        self.batch_size = max(1, min(64, 4_000_000 // self.x.size))

    def predictions(self, masks):
        enabled = np.zeros((len(masks), self.physical_width), dtype=bool)
        for i, mask in enumerate(masks):
            positions = self.positions[np.asarray(mask, dtype=np.intp)]
            enabled[i, positions[positions >= 0]] = True
        data = np.where(enabled[:, None, :], self.x, self.fill)
        predict = self.model.predict if self.regression else self.model.predict_proba
        pred = np.asarray(predict(data.reshape(-1, self.physical_width)), dtype=float)
        return pred.reshape(len(masks), len(self.x), *pred.shape[1:])

    def utilities(self, masks, labels, weights):
        preds = self.predictions(masks)
        if self.regression:
            return [max(0., float(r2_score(labels, p, sample_weight=weights))) for p in preds]
        return [weighted_auc(labels, p, weights) for p in preds]
