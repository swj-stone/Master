"""Compare paired times and exact predictions in two new processes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(quality,first,second,output):
    output.mkdir(parents=True,exist_ok=False)
    runs=(quality,first,second)
    states=[json.loads((p/'protocol.json').read_text()) for p in runs]
    assert [s.pop('pass_id') for s in states]==[0,1,2]
    assert states[0]==states[1]==states[2]
    counts=0
    for panel in states[0]['panels']:
        file=panel['panel']+'.npz'
        with np.load(quality/file) as ref, np.load(first/file) as one, np.load(second/file) as two:
            assert set(ref.files)==set(one.files)==set(two.files)
            for field in ref.files:
                np.testing.assert_array_equal(one[field],ref[field])
                np.testing.assert_array_equal(two[field],ref[field])
            counts+=len(ref.files)
    raw=pd.concat([pd.read_csv(p/'trials.csv').assign(run=i) for i,p in enumerate((first,second),1)])
    keys=['panel','run','method']
    grouped=raw.groupby(keys)
    assert (grouped.size()==3).all()
    for key in ('k','score'):assert (grouped[key].nunique()==1).all()
    medians=grouped[['total_ms','select_ms','k','score']].median().reset_index()
    medians.to_csv(output/'medians.csv',index=False)
    distribution=pd.read_csv(quality/'random-quality.csv').groupby('panel').score
    assert (distribution.count()==20).all()
    comparisons=[]
    for panel,g in medians.groupby('panel'):
        baseline=g.pivot(index='method',columns='run',values='total_ms')
        scores=g.groupby('method').score.first()
        k=g.groupby('method').k.first()
        assert g.groupby('method').score.nunique().eq(1).all()
        details=states[0]['panels'][[x['panel'] for x in states[0]['panels']].index(panel)]
        row=dict(panel=panel,source=panel.rsplit('-',2)[0],p=details['p'],k=int(k['conditional']),
                 domain=details['target_domain'],score=scores['conditional'],
                 random_median=distribution.median()[panel],random_wins=int((distribution.get_group(panel)<scores['conditional']).sum()))
        for method in baseline.index:
            row['delta_'+method]=scores['conditional']-scores[method]
            row['ratio_'+method]=max(baseline.loc['conditional']/baseline.loc[method])
        row['quality']=bool(row['delta_full'] >= -.02-1e-12)
        row['joint']=row['quality'] and row['k']<row['p'] and row['ratio_full']<1
        row['joint_and_both']=row['joint'] and row['delta_F']>1e-12 and row['score']>row['random_median']+1e-12
        comparisons.append(row)
    pairs=pd.DataFrame(comparisons)
    pairs.to_csv(output/'pairs.csv',index=False)
    bysource=pairs.groupby('source').agg(n=('panel','size'),width=('k','mean'),score=('score','mean'),
         delta_F=('delta_F','mean'),delta_zero=('delta_zero','mean'),delta_grouped=('delta_grouped','mean'),
         median_time_ratio_F=('ratio_F','median'),median_time_ratio_zero=('ratio_zero','median'),
         median_time_ratio_grouped=('ratio_grouped','median'),median_time_ratio_full=('ratio_full','median'),
         joint=('joint','sum'),joint_and_both=('joint_and_both','sum'))
    bysource.to_csv(output/'sources.csv')
    print(bysource.to_string())
    (output/'verification.json').write_text(json.dumps(dict(trials=len(raw),panels=len(pairs),arrays_per_process=counts,
       quality_and_timed_outputs_identical=True,kept_full_success=False,accepted_domain_sources=[
       str(src) for src,g in pairs[pairs.domain].groupby('source') if g.joint_and_both.mean()>.5]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('quality','first','second','output'):parser.add_argument(key,type=Path)
    a=parser.parse_args();compare(a.quality,a.first,a.second,a.output)
