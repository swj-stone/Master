"""Train-only word-presence folds from the original Turkish SMS collection."""
from collections import Counter
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

from experiments.document_vote_benchmark import ROOT, SEEDS, SPLIT_SEED, csr_row_keys, digest, file_hash


ASSET = ROOT / "data/raw/turkish-sms-original.csv"
SHA256 = "fe4e9b1e7f4d2a6281a89f864b730e1fc7cd894ed29a816ecbf037b048c7025d"
SOURCE_URL = "https://github.com/onrkrsy/TurkishSMS-Collection"


def panels():
    if file_hash(ASSET) != SHA256:
        raise ValueError("Turkish SMS archive digest mismatch")
    data = pd.read_csv(ASSET, sep=";")
    assert set(data.columns) == {"Message", "Group", "GroupText"}
    assert data.Message.notna().all() and set(data.Group) == {1, 2}
    documents = data.Message.astype(str).to_numpy()
    labels = data.Group.eq(1).to_numpy(dtype=int)
    analyze = CountVectorizer().build_analyzer()
    bags = [tuple(sorted(set(analyze(text)))) for text in documents]
    counts = Counter(bags)
    selected = np.array([i for i, bag in enumerate(bags) if counts[bag] == 1], dtype=np.intp)
    docs, y = documents[selected], labels[selected]
    splitter = StratifiedKFold(3, shuffle=True, random_state=SPLIT_SEED)
    for fold, (training, test) in enumerate(splitter.split(docs, y), 1):
        start = perf_counter_ns()
        vectorizer = CountVectorizer(binary=True, dtype=np.float64)
        X = vectorizer.fit_transform(docs[training])
        Xt = vectorizer.transform(docs[test])
        X.sort_indices()
        Xt.sort_indices()
        frequency = np.bincount(X.indices, minlength=X.shape[1])
        varying = (frequency > 0) & (frequency < len(training))
        X, Xt = X[:, varying], Xt[:, varying]
        X.sort_indices()
        Xt.sort_indices()
        known = set(csr_row_keys(X))
        new = np.array([key not in known for key in csr_row_keys(Xt)])
        Xt = Xt[new]
        elapsed = (perf_counter_ns() - start) / 1e6
        freq = np.bincount(X.indices, minlength=X.shape[1])
        assert new.any() and np.all((freq > 0) & (freq < X.shape[0]))
        assert set(y[training]) == set(y[test[new]]) == {0, 1}
        info = dict(
            dataset="turkish_sms", fold=fold, seed=SEEDS[fold-1], suite="external", stage="external",
            task="spam", n_train=len(training), n_test=int(new.sum()), p=X.shape[1],
            train_spam=int(y[training].sum()), test_spam=int(y[test[new]].sum()),
            source_records=len(data), drop_duplicate_bags=len(data)-len(selected),
            train_vocabulary=len(varying), drop_constant=int((~varying).sum()),
            drop_cross_test=int((~new).sum()), common_prepare_ms=elapsed,
            train_ids_sha256=digest(selected[training]), test_ids_sha256=digest(selected[test[new]]),
            vocabulary_sha256=digest(np.array(vectorizer.get_feature_names_out()[varying], dtype="U")),
            train_rows_sha256=digest(X.indptr), train_cols_sha256=digest(X.indices),
            test_rows_sha256=digest(Xt.indptr), test_cols_sha256=digest(Xt.indices),
        )
        yield X, y[training], Xt, y[test[new]], info
