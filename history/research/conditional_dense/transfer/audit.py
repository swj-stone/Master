"""Audit saved predictions, timed supports and paired source/model differences."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from research.conditional_dense.transfer.study import METHODS, MODELS, ROOT


def score(truth,pred,units):
    if units is None:return balanced_accuracy_score(truth,pred)
    return np.mean([balanced_accuracy_score(truth[units==u],pred[units==u]) for u in np.unique(units)])


def summarize(rawdir,output):
    output.mkdir(parents=True,exist_ok=False)
    rows=pd.read_csv(rawdir/'outcomes.csv')
    ids=['source','fold','seed','model','method']
    assert len(rows)==192 and not rows.duplicated(ids).any()
    assert set(rows.method)==set(METHODS) and set(rows.model)==set(MODELS)
    np.testing.assert_allclose(rows[['select_ms','encode_ms','fit_ms','predict_ms']].sum(axis=1),
                               rows.total_ms,atol=1e-7,rtol=1e-9)
    pairs=[]
    for (source,fold,seed),cohort in rows.groupby(ids[:3]):
        key=f'{source}-{fold}-{seed}'
        old='conditional-speech-quality-20261118a' if source=='spoken_arabic_digit' else 'conditional-dense-quality-20261118a'
        with np.load(rawdir/(key+'.npz')) as store,np.load(ROOT/'results/minpi-study'/old/(key+'.npz')) as frozen:
            for arm in METHODS:np.testing.assert_array_equal(store[arm+'_support'],frozen[arm+'_cols'])
            units=store['units'] if 'units' in store else None
            assert (units is not None)==(source=='har')
            np.testing.assert_array_equal(store['truth'],frozen['truth'])
            for model,group in cohort.groupby('model'):
                compare=group.set_index('method'); own=compare.loc['conditional']
                pair=dict(source=source,fold=fold,seed=seed,model=model,p=own.p,k=own.k,
                          encoded_k=own.encoded_k,score=own.score,total_ms=own.total_ms)
                for arm in METHODS:
                    ref=compare.loc[arm]
                    cols=store[arm+'_support']
                    assert cols.size==len(np.unique(cols))==ref.k
                    fresh=score(store['truth'],store[model+'_'+arm+'_pred'],units)
                    assert abs(fresh-ref.score)<1e-12
                    pair['score_'+arm]=ref.score
                    pair['delta_'+arm]=own.score-ref.score
                    pair['ms_'+arm]=ref.total_ms
                    pair['ratio_'+arm]=own.total_ms/ref.total_ms
                    pair['k_'+arm]=ref.k
                pairs.append(pair)
    results=pd.DataFrame(pairs).sort_values(['source','fold','seed','model'])
    results.to_csv(output/'pairs.csv',index=False)
    summarize=[]
    for (source,model),g in results.groupby(['source','model']):
        row=dict(source=source,model=model,n=len(g),k_min=int(g.k.min()),k_max=int(g.k.max()),
                 score=g.score.mean(),ms_mean=g.total_ms.mean())
        for arm in METHODS:
            row['score_'+arm]=g['score_'+arm].mean()
            row['delta_'+arm]=g['delta_'+arm].mean()
            row['median_ratio_'+arm]=g['ratio_'+arm].median()
            row['strict_wins_'+arm]=int((g['delta_'+arm]>1e-12).sum())
            row['quality_and_speed_wins_'+arm]=int(((g['delta_'+arm]>1e-12)&(g['ratio_'+arm]<1)).sum())
        row['min_full_delta']=g.delta_full.min()
        row['max_full_delta']=g.delta_full.max()
        row['full_faster']=int((g.ratio_full<1).sum())
        row['historic_joint_count']=int(((g.ratio_full<1)&(g.k<g.p)&(g.delta_full>=-.02-1e-12)).sum())
        summarize.append(row)
    report=pd.DataFrame(summarize)
    report.to_csv(output/'source-model.csv',index=False)
    convergence=rows.groupby(['model','method']).converged.agg(['size','sum'])
    convergence.columns=['runs','converged'];convergence.to_csv(output/'convergence.csv')
    (output/'audit.json').write_text(json.dumps(dict(inputs=8,model_cases=len(results),raw_pipelines=len(rows),
        matched_frozen_supports=8*len(METHODS),rescored_predictions=len(rows),
        total_equals_sum_of_components=True),indent=2))
    print(report[['source','model','score','delta_full','delta_F','delta_F_half','delta_zero',
                  'median_ratio_full','median_ratio_F','median_ratio_F_half','strict_wins_F','full_faster']].round(4).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rawdir',type=Path);parser.add_argument('output',type=Path)
    a=parser.parse_args();summarize(a.rawdir,a.output)
