"""Training-only mechanism development. Never score the outer test split here."""
import csv
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from experiments.document_vote_benchmark import panels, fit_predict
from experiments import fast_document_vote_benchmark as old
from research.text_peer.efficient import rank_words as support


def main():
    rows = []
    for X, y, _, _, meta in panels():
        if meta["fold"] != 1 or meta["dataset"] not in ("sms","amazon_cells","imdb","yelp"):
            continue
        for split, (a,b) in enumerate(StratifiedShuffleSplit(n_splits=3,test_size=.25,random_state=60291).split(X,y)):
            train, test, labels = X[a], X[b], y[a]
            for norm in (0.,1.,1.645):
                selected = support(train, labels,penalty=norm)
                width = len(selected)
                for method in ("candidate","zero","all","F_match","pooled_match","random_match","fast_vote","full"):
                    if method in ("zero","all"):
                        cols=support(train,labels,count=width,rule=method,penalty=norm)
                    elif method=="candidate":
                        cols=selected
                    else:
                        cols=old.select(method,train,labels,1129,width)
                    pred=fit_predict(train,labels,test,cols,"classification",1129,False)
                    rows.append(dict(dataset=meta["dataset"],inner_split=split,normalization=norm,arm=method,width=len(cols),score=balanced_accuracy_score(y[b],pred)))
    path=Path(__file__).with_name("development-penalties.csv")
    with path.open("x",newline="") as fp:
        out=csv.DictWriter(fp,fieldnames=rows[0]);out.writeheader();out.writerows(rows)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
