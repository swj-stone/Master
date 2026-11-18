"""Independent original Turkish mail folds; no candidate-dependent processing."""
from collections import Counter
import csv
import io
import re
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

from experiments.document_vote_benchmark import ROOT, SEEDS, SPLIT_SEED, csr_row_keys, digest, file_hash


ASSET = ROOT / "data/raw/turkish-mail-v01.zip"
SHA256 = "3d3c7e4254e607059a901b47bf9cac1a06429bb0b7bbfad5e22e80f84c0b6e29"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/530/turkish+spam+v01"


def corpus():
    """Retain every labeled record and audit the publisher's malformed CSV."""
    if file_hash(ASSET) != SHA256:
        raise ValueError("Turkish mail archive digest mismatch")
    with ZipFile(ASSET) as archive:
        assert archive.namelist() == ["trspam.csv"]
        raw = archive.read("trspam.csv")
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    header, content = text.split("\n", 1)
    assert header.rstrip("\r") == "Text,Classification"
    pieces = re.split(r",(spam|ham)\r?\n", content)
    assert pieces[-1] == ",,496\r\n", "Unexpected unlabeled trailing content"
    documents, labels = [], []
    for body, label in zip(pieces[:-1:2], pieces[1::2]):
        if body.startswith('"') and body.endswith('"'):
            body = body[1:-1].replace('""', '"')
        documents.append(body)
        labels.append(int(label == "spam"))
    assert len(documents) == 825 and sum(labels) == 329
    # csv.reader independently respects multiline quoted fields. Unescaped
    # body quotes produce extra comma fields; their join preserves token bags.
    parsed = list(csv.reader(io.StringIO(text)))
    assert parsed[0] == ["Text", "Classification"] and parsed[-1] == ["", "", "496"]
    records = parsed[1:-1]
    assert len(records) == len(documents)
    analyze = CountVectorizer().build_analyzer()
    for body, label, row in zip(documents, labels, records):
        assert row[-1] in ("spam", "ham") and int(row[-1] == "spam") == label
        assert set(analyze(body)) == set(analyze(",".join(row[:-1])))
    metadata = dict(source_records=825, source_spam=329, source_ham=496,
                    advertised_records=826, excluded_summary_rows=1,
                    source_empty_text=sum(not body.strip() for body in documents),
                    csv_reader_token_bag_agreement=825)
    return np.asarray(documents), np.asarray(labels, dtype=int), metadata


def panels():
    documents, labels, source = corpus()
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
            dataset="turkish_mail", fold=fold, seed=SEEDS[fold-1], suite="external", stage="external",
            task="spam", n_train=len(training), n_test=int(new.sum()), p=X.shape[1],
            train_spam=int(y[training].sum()), test_spam=int(y[test[new]].sum()),
            **source, drop_duplicate_bags=len(documents)-len(selected),
            train_vocabulary=len(varying), drop_constant=int((~varying).sum()),
            drop_cross_test=int((~new).sum()), common_prepare_ms=elapsed,
            train_ids_sha256=digest(selected[training]), test_ids_sha256=digest(selected[test[new]]),
            vocabulary_sha256=digest(np.array(vectorizer.get_feature_names_out()[varying], dtype="U")),
            train_rows_sha256=digest(X.indptr), train_cols_sha256=digest(X.indices),
            test_rows_sha256=digest(Xt.indptr), test_cols_sha256=digest(Xt.indices),
        )
        yield X, y[training], Xt, y[test[new]], info
