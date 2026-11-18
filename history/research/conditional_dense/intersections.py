"""Same-configuration intersections across the main and additional speech study."""
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parent


def summary():
    frames=[pd.read_csv(BASE/folder/'pairs.csv') for folder in ('evidence','speech-evidence')]
    pairs=pd.concat(frames,ignore_index=True)
    ordered=[]
    for src,g in pairs.groupby('source'):
        joint=g.joint
        f=(g.delta_F>1e-12)&(g.ratio_F<1)
        z=g.delta_zero>1e-12
        independent=(g.delta_F_half>1e-12)&(g.ratio_F_half<1)&(g.k<=(g.p+1)//2)
        ordered.append(dict(source=src,n=len(g),width_min=g.k.min(),width_max=g.k.max(),
            F_half_width=int((g.p.iloc[0]+1)//2),full_joint=int(joint.sum()),
            joint_F_quality_cost=int((joint&f).sum()),joint_F_quality_cost_zero_quality=int((joint&f&z).sum()),
            joint_F_and_zero_both_quality_cost=int((joint&f&z&(g.ratio_zero<1)).sum()),
            joint_independent_F_no_wider_quality_cost=int((joint&independent).sum()),
            joint_F_quality_random_median=int(g.joint_and_both.sum())))
    return pd.DataFrame(ordered)

if __name__=='__main__':print(summary().to_csv(index=False),end='')
