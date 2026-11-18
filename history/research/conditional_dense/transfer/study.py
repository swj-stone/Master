"""Transfer the frozen dense minPI supports through four final learners."""
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from experiments.compare_predictive_minpi import choose
from experiments.minpi_backward_confirmation import evaluate
from experiments.minpi_stable_shift import get_shift
from research.conditional_dense.confirmation import Panel, SEEDS, ROOT, deploy, zero_vote
from research.conditional_dense.shared_method import conditional_support
from research.conditional_dense.speech import ARCHIVE, SHA256, utterances
from research.four_stage.benchmark import MODELS, encode

METHODS=('conditional','full','F','random','F_half','zero')


def panels():
    for i,(seed,X,y,_,Xt,truth,_,_,_) in enumerate(deploy.configurations('isolet'),1):
        if i>1: yield Panel('isolet',i,seed,X,y,Xt,truth,None)
    X,y,groups,Xt,truth,units,_=get_shift('har')
    assert not np.intersect1d(groups,units).size
    for seed in SEEDS: yield Panel('har',1,seed,X,y,Xt,truth,units)
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==SHA256
    with ZipFile(ARCHIVE) as archive:
        X,y=utterances(archive.read('Train_Arabic_Digit.txt'),660)
        Xt,truth=utterances(archive.read('Test_Arabic_Digit.txt'),220)
    for seed in SEEDS: yield Panel('spoken_arabic_digit',1,seed,X,y,Xt,truth,None)


def selection(panel,method,width):
    if method=='conditional': return conditional_support(panel.X,panel.y,seed=panel.seed)[0]
    if method=='zero': return zero_vote(panel.X,panel.y,width,panel.seed)
    if method in ('full','F','random','F_half'):
        k=(panel.X.shape[1]+1)//2 if method=='F_half' else width
        return choose(method.removesuffix('_half').lower(),panel.X,panel.y,'classification',panel.seed,k)
    raise ValueError(method)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in ('research/conditional_dense/transfer/study.py','research/conditional_dense/shared_method.py',
                 'research/four_stage/benchmark.py','experiments/compare_predictive_minpi.py',
                 'research/conditional_dense/confirmation.py','research/conditional_dense/speech.py'):
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    protocol=dict(data=['isolet','har','spoken_arabic_digit'],models=MODELS,methods=METHODS,
                  seed=SEEDS,threads=1,scaling='train-only encode, float32',
                  MLP=dict(hidden=32,epochs=40,batch_size=128,learning_rate=.003),
                  timing='one fresh pipeline per configuration, descriptive non-isolated timing',
                  sklearn=sklearn.__version__,numpy=np.__version__,source_sha256=hashes)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with (output/'outcomes.csv').open('x',newline='') as fp:
        writer=csv.writer(fp)
        writer.writerow(('source','fold','seed','model','method','p','k','encoded_k','score','select_ms',
                         'encode_ms','fit_ms','predict_ms','total_ms','converged'))
        for panel in panels():
            runname='conditional-speech-quality-20261118a' if panel.name.startswith('spoken') else 'conditional-dense-quality-20261118a'
            key=f'{panel.name}-{panel.fold}-{panel.seed}'
            archive=ROOT/'results/minpi-study'/runname/(key+'.npz')
            with np.load(archive) as saved:
                np.testing.assert_array_equal(panel.gold,saved['truth'])
                original={arm:saved[arm+'_cols'].copy() for arm in METHODS}
            width=len(original['conditional'])
            store=dict(truth=panel.gold, **{arm+'_support':cols for arm,cols in original.items()})
            if panel.units is not None:store['units']=panel.units
            rng=np.random.default_rng([panel.seed,panel.fold,10519])
            for model in MODELS:
                for method in rng.permutation(METHODS):
                    start=perf_counter()
                    cols=selection(panel,method,width)
                    selected=perf_counter()
                    if len(cols)==panel.X.shape[1]: A,B=panel.X,panel.Xt
                    else: A,B=panel.X[:,cols],panel.Xt[:,cols]
                    A,B=encode(A,B,model.endswith('_ple'))
                    encoded=perf_counter()
                    if model.startswith('ridge'):
                        est=RidgeClassifier(alpha=1.,solver='cholesky')
                    else:
                        est=MLPClassifier(hidden_layer_sizes=(32,),max_iter=40,random_state=panel.seed,
                                          learning_rate_init=.003,batch_size=min(128,len(A)))
                    with warnings.catch_warnings(record=True) as messages:
                        warnings.simplefilter('always',ConvergenceWarning)
                        est.fit(A,panel.y)
                    fitted=perf_counter()
                    pred=est.predict(B)
                    end=perf_counter()
                    np.testing.assert_array_equal(cols,original[method])
                    score,_=evaluate(panel.name,panel.gold,pred,panel.units,'classification')
                    writer.writerow((panel.name,panel.fold,panel.seed,model,method,panel.X.shape[1],len(cols),
                        A.shape[1],score,1000*(selected-start),1000*(encoded-selected),1000*(fitted-encoded),
                        1000*(end-fitted),1000*(end-start),not any(issubclass(m.category,ConvergenceWarning) for m in messages)))
                    store[model+'_'+method+'_pred']=pred
                fp.flush()
            np.savez_compressed(output/(key+'.npz'),**store)
            print(key,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    with threadpool_limits(1):run(parser.parse_args().output)
