"""Check physical-unit votes and production CSV inference against direct traversal."""
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.minpi_rework import predictive_rf_deploy as cli, quota_coalition_rf as rf
from experiments.minpi_rework.predictive_mask_study import centered
from minppi_mofs.grouped_paths import collapse_path_groups
from minppi_mofs.predictive_quota import PredictiveGame
from minppi_mofs.predictive_rf_artifact import RestrictedRF


def main():
    rng = np.random.default_rng(382)
    x = rng.normal(size=(260, 6))
    units = np.array([2, 0, 1, 2, 1, 0])
    names = [f"channel_{i}" for i in range(len(units))]
    unit_names = [f"device_{i}" for i in units]
    y = x[:,0] + x[:,3] - .5*x[:,2]
    for task, truth in (("classification", np.digitize(y,[-.5,.5])), ("regression", y)):
        forest = rf.fit(x[:100], truth[:100], task, 44, n_estimators=20)
        visible = x[100:130]
        target = truth[100:130]
        direct_target = (np.eye(3)[target] if task == "classification" else target[:,None])
        raw = forest.game(visible)
        a, incidence = collapse_path_groups(raw.a, raw.incidence, units)
        game = PredictiveGame.from_paths(a, incidence, centered(forest,target,task))
        masks = [frozenset(mask) for n in range(4) for mask in combinations(range(3), n)]
        losses = {}
        for mask in masks:
            features = np.flatnonzero(np.isin(units,list(mask)))
            pred = forest.predict(visible[:,features],features).reshape(direct_target.shape)
            losses[mask] = np.square(pred-direct_target).sum()
        for support in masks:
            expected = [np.mean([losses[t & support]-losses[t] for t in masks if len(t)==s]) for s in range(4)]
            np.testing.assert_allclose(game.conditional_risk(np.isin(np.arange(3),list(support))),
                                       expected,atol=1e-11,rtol=1e-11)

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, rows in (("fit",slice(0,100)),("vote",slice(100,160)),("accept",slice(160,220))):
                data = pd.DataFrame(x[rows],columns=names)
                data["target"] = truth[rows]
                data.to_csv(tmp/f"{name}.csv",index=False)
            pd.DataFrame({"feature":names,"unit":unit_names}).iloc[::-1].to_csv(tmp/"units.csv",index=False)
            cli.main(["fit","--fit",str(tmp/"fit.csv"),"--vote",str(tmp/"vote.csv"),
                      "--accept",str(tmp/"accept.csv"),"--target","target","--task",task,
                      "--groups",str(tmp/"units.csv"),"--reachable-only","--model",str(tmp/"forest.npz")])
            model = RestrictedRF.load(tmp/"forest.npz")
            assigned = model.metadata["acquisition_units"]
            assert all(assigned[n] == unit_names[names.index(n)] for n in model.names)
            X = pd.DataFrame(x[220:],columns=names)
            availability = pd.DataFrame(rng.integers(2,size=(len(X),3)),columns=[f"device_{i}" for i in range(3)])
            X.to_csv(tmp/"test.csv",index=False)
            availability.to_csv(tmp/"availability.csv",index=False)
            cli.main(["predict","--model",str(tmp/"forest.npz"),"--input",str(tmp/"test.csv"),
                      "--unit-availability",str(tmp/"availability.csv"),"--output",str(tmp/"out.csv")])
            flags = availability[[assigned[n] for n in model.names]].to_numpy(bool)
            expected = model.predict(X[model.names].to_numpy(),flags)
            actual = pd.read_csv(tmp/"out.csv")
            if task == "classification":
                columns = ["probability_"+name for name in model.classes]
                actual = actual[columns].to_numpy()
            else:
                actual = actual.prediction.to_numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
    print("All 8 unit supports, 8 availability patterns and both deployed CSV interfaces agree.")


if __name__ == "__main__":
    main()
