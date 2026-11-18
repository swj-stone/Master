"""Produce descriptive split and dataset tables from frozen formal minPI runs."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.minpi_study import OUTPUT, P


def jaccard(a, b):
    return len(a & b) / len(a | b)


def main():
    rows, overlap, stability = [], [], []
    groups = {}
    for name in P['datasets']:
        runs = list((OUTPUT/'formal'/name).glob('s*/record.json'))
        if len(runs) != len(P['formal_seeds']):
            raise ValueError(f'{name}: incomplete, {len(runs)} cases')
        for path in sorted(runs):
            record = json.loads(path.read_text(encoding='utf-8'))
            arms = record['arms']
            seed = record['seed']
            if set(arms) != set(P['arms']):
                raise ValueError(f'{name}/{seed}: arm set differs from protocol')
            truth = set(record['known_signal_original_ids'])
            original = record['original_column_indices']
            exact = set(arms['exact']['selected'])
            with np.load(path.parent/'predictions.npz') as saved:
                rates = saved['exact_rates']
                mc_score = saved['mc32_scores']
                score = rates.min(0)
                # Largest e.g. mask fraction is measured on the full ranking, including ties.
                counts = np.isclose(rates,score[None,:],rtol=0,atol=1e-12).sum(axis=0)
                info = dict(dataset=name, seed=seed, p=record['p'], k=len(exact),
                            maximum_mc_abs_error=np.max(np.abs(mc_score-score)),
                            mean_mc_abs_error=np.mean(np.abs(mc_score-score)),
                            minimum_tied_fraction=np.mean(counts>1))
                for i, tier in enumerate(arms['exact']['diagnostics']['tiers']):
                    info[f'fraction_strict_min_k{tier}'] = np.mean((counts==1)&(np.abs(rates[i]-score)<=1e-12))
                for method in ('mean','vote','mc32','k16','f','margin','legacy_compact_rank'):
                    info[f'jaccard_{method}'] = jaccard(exact,set(arms[method]['selected']))
                overlap.append(info)
            for method, arm in arms.items():
                support = {original[j] for j in arm['selected']}
                groups.setdefault((name,method), []).append(support)
                for learner, result in arm['backends'].items():
                    metric = 'auc' if record['task']=='classification' else 'r2'
                    full = arms['full']['backends'][learner]
                    fbase = arms['f']['backends'][learner]
                    rows.append(dict(dataset=name, seed=seed, task=record['task'], learner=learner,
                                method=method, primary_metric=metric, primary=result[metric],
                                n=record['rows'],p=record['p'],k=arm['k'],retained_fraction=arm['k']/record['p'],
                                selector_seconds=arm['selector_seconds'], **result,
                                delta_vs_full=result[metric]-full[metric], delta_vs_f=result[metric]-fbase[metric],
                                time_ratio_vs_full=result['total_seconds']/full['total_seconds'],
                                time_ratio_vs_f=result['total_seconds']/fbase['total_seconds'],
                                known_signal_recall=(len(support&truth)/len(truth) if truth else None)))
    for (name,method), supports in groups.items():
        pairs=[jaccard(a,b) for a,b in combinations(supports,2)]
        stability.append(dict(dataset=name,method=method,pairs=len(pairs),mean_jaccard=np.mean(pairs),
                              minimum_jaccard=min(pairs),maximum_jaccard=max(pairs)))
    folder=OUTPUT/'summary'
    folder.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows).sort_values(['dataset','seed','learner','method'])
    frame.to_csv(folder/'split_metrics.csv',index=False)
    cols=['n','p','k','primary','accuracy','balanced_accuracy','mae','selector_seconds','total_seconds',
          'delta_vs_full','delta_vs_f','time_ratio_vs_full','time_ratio_vs_f','known_signal_recall']
    frame.groupby(['dataset','task','learner','method'])[cols].median().to_csv(folder/'dataset_medians.csv')
    pd.DataFrame(overlap).to_csv(folder/'ablation_overlap.csv',index=False)
    pd.DataFrame(stability).to_csv(folder/'support_stability.csv',index=False)
    pairs=[]
    for name in P['datasets']:
        part=frame[frame.dataset==name]
        for learner in P['learners']:
            panel=part[part.learner==learner].set_index(['method','seed']).sort_index()
            for method,other in [('exact',m) for m in P['arms'] if m!='exact']+[('exact_forest','forest')]:
                a=panel.loc[method]
                b=panel.loc[other]
                d=a.primary-b.primary
                ratio=a.total_seconds/b.total_seconds
                secondary=a.accuracy-b.accuracy
                tolerance=P['quality_margins'][a.primary_metric.iloc[0]]
                ok=(d>=-tolerance)&(secondary.isna() | (secondary>=-P['quality_margins']['accuracy']))
                fast=(ratio<=P['paired_time_ratio_for_speed_claim']) & (a.k<=b.k)
                pairs.append(dict(dataset=name,learner=learner,method=method,reference=other,splits=len(d),
                             primary_difference_median=d.median(),primary_difference_min=d.min(),
                             primary_difference_max=d.max(),accuracy_difference_median=secondary.dropna().median()
                             if secondary.notna().any() else None,
                             time_ratio_median=ratio.median(),selector_ratio_median=(a.selector_seconds/b.selector_seconds).median(),
                             quality_pass_splits=int(ok.sum()),quality_and_speed_pass_splits=int((ok&fast).sum())))
    pd.DataFrame(pairs).to_csv(folder/'paired_comparisons.csv',index=False)
    print(f'{len(frame)} backend rows across {len(overlap)} splits -> {folder}')


if __name__=='__main__':
    main()

