"""Sensitivity of the polarity experiment to agreement in the nuisance background."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.minpi_study_calibration import world
from experiments.minpi_study_mechanism import METHODS, ROOT, TIERS, WORLDS


POLICIES = (.5, 2/3, 5/6, 1.)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    all_rows, contrasts = [], []
    indices = np.random.default_rng(20261122).integers(WORLDS, size=(10000, WORLDS))
    with threadpool_limits(limits=1):
        for policy_index, match in enumerate(POLICIES):
            rows, observations = [], []
            for i in range(WORLDS):
                records, data = world(i, entropy=20261121, background_match=match)
                rows.extend(records)
                observations.append(data)
            table = pd.DataFrame(rows)
            table['background_match'] = match
            all_rows.append(table)
            for rhs in ('largest', 'sign'):
                for metric in ('worst_layer', 'worst_instrument_auc', 'stable_count', 'auc', 'logloss'):
                    paired = table.pivot(index='world', columns='method', values=metric)
                    delta = (paired['minimum'] - paired[rhs]).to_numpy()
                    primary = policy_index == 2 and metric in ('worst_layer', 'worst_instrument_auc')
                    alpha = .0125 if primary else .05
                    low, high = np.quantile(delta[indices].mean(axis=1), [alpha/2, 1-alpha/2])
                    contrasts.append(dict(background_match=match, contrast=f'minimum-{rhs}',
                                          metric=metric, mean=delta.mean(), low=low, high=high,
                                          coverage=1-alpha, worlds=WORLDS))
            np.savez_compressed(output / f'policy-{policy_index}.npz', method_order=METHODS,
                                tiers=TIERS, background_match=match,
                                **{k: [x[k] for x in observations] for k in observations[0]})
    pd.concat(all_rows, ignore_index=True).to_csv(output / 'worlds.csv', index=False)
    pd.DataFrame(contrasts).to_csv(output / 'contrasts.csv', index=False)
    sources = (ROOT / 'docs/minpi-study/acquisition-bias-protocol.md', Path(__file__),
               ROOT / 'experiments/minpi_study_calibration.py')
    digests = [f'{p.name} {sha256(p.read_bytes()).hexdigest()}' for p in sources]
    (output / 'readme.txt').write_text('\n'.join(digests) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/minpi-study/mechanism/acquisition-bias')
    main(parser.parse_args().output)
