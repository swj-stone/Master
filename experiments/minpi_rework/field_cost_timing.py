"""Time a warmed full twenty-arm acquisition benchmark replay against sealed outputs."""
from io import BytesIO
import json
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_rework.field_cost import OUT, collect_fit


def main():
    receipts = OUT/"forecast-receipts.zip"
    times = []
    rng = np.random.default_rng(6972)
    with ZipFile(receipts) as z:
        collect_fit(12,42)
        for repeat in range(3):
            for slot in rng.permutation(10):
                p,seed = 12+slot//5, 42+slot%5
                p,seed = int(p),int(seed)
                tick = perf_counter()
                info, preds = collect_fit(p,seed)
                elapsed = perf_counter()-tick
                key = f"p{p}-{seed}"
                assert info == json.loads(z.read(key+".json"))
                with np.load(BytesIO(z.read(key+".npz"))) as original:
                    for arm, pred in preds.items():
                        np.testing.assert_array_equal(pred, original[arm])
                times.append(dict(p=p,seed=seed,repeat=repeat,all_arm_s=elapsed,
                    eligible_universe=1<<p,output_arms=len(preds)))
    pd.DataFrame(times).to_csv(OUT/"replay-timings.csv",index=False)
    print(pd.DataFrame(times).groupby("p").all_arm_s.agg(["median","min","max"]))


if __name__=="__main__":
    with threadpool_limits(1):
        main()
