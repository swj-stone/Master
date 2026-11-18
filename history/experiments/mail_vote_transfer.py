"""External acquisition-batch transfer of the unchanged document minPI and its controls."""
import argparse
import json
import re
import tarfile
from collections import Counter
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import (
    ARMS, ROOT, SEEDS, csr_row_keys, digest, file_hash, run_case, summarize, write_json,
)


# The first three archives form the training batch. The final two form the test batch.
ARCHIVES = {
    "easy_ham": "2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438",
    "hard_ham": "ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9",
    "spam": "c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743",
    "easy_ham_2": "b4bd3dc5ae5b40f38e99a0e41ad7d16b428b56e818e3a6ffa07330d62004dd38",
    "spam_2": "40bc444c596ff3e71505ae334f115b16f9b946786ee68f09b921b6b99217fd00",
}


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def body(raw):
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    payload = part.get_payload(decode=True) or b""
    try:
        text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    except LookupError:
        text = payload.decode("utf-8", errors="replace")
    if part.get_content_type() == "text/html":
        parser = VisibleText()
        parser.feed(text)
        text = " ".join(parser.parts)
    return text


def prepare():
    analyze = CountVectorizer().build_analyzer()
    records = []
    for folder, checksum in ARCHIVES.items():
        path = ROOT / "data/raw" / f"20030228_{folder}.tar.bz2"
        if file_hash(path) != checksum:
            raise ValueError(f"Archive digest mismatch for {folder}")
        with tarfile.open(path, "r:bz2") as archive:
            for item in sorted(archive.getmembers(), key=lambda item: item.name):
                name = Path(item.name).name
                if not item.isfile() or not re.fullmatch(r"[0-9]+\.[0-9a-f]{32}", name):
                    continue
                if folder == "spam_2" and name == "00529.0c8a07bb7b14576063ba0c1c4079e209":
                    continue
                tokens = sorted(set(analyze(body(archive.extractfile(item).read()))))
                records.append((folder, name, " ".join(tokens)))
    counts = Counter(text for _, _, text in records)
    rows = [(folder, name, text) for folder, name, text in records if text and counts[text] == 1]
    train = [r for r in rows if not r[0].endswith("_2")]
    test = [r for r in rows if r[0].endswith("_2")]
    start = perf_counter()
    vectorizer = CountVectorizer(binary=True, dtype=np.float64)
    X = vectorizer.fit_transform(r[2] for r in train)
    Xt = vectorizer.transform(r[2] for r in test)
    freq = np.bincount(X.indices, minlength=X.shape[1])
    keep = (freq > 0) & (freq < X.shape[0])
    X, Xt = X[:, keep], Xt[:, keep]
    X.sort_indices()
    Xt.sort_indices()
    keys = set(csr_row_keys(X))
    held = np.array([k not in keys for k in csr_row_keys(Xt)])
    Xt = Xt[held]
    elapsed = (perf_counter() - start) * 1000
    y = np.array([r[0].startswith("spam") for r in train], dtype=int)
    labels = np.array([r[0].startswith("spam") for r in test], dtype=int)[held]
    assert set(y) == set(labels) == {0, 1}
    assert min(X.shape) > 10 and len(train) > 2000 and len(labels) > 1800
    meta = dict(dataset="spamassassin", fold=1, task="spam", n_train=len(y),
                n_test=len(labels), train_spam=int(y.sum()), test_spam=int(labels.sum()),
                p=X.shape[1], source_records=len(records), empty=sum(not text for _, _, text in records),
                drop_duplicate_bags=sum(bool(t) and counts[t] > 1 for _, _, t in records),
                drop_constant=int((~keep).sum()), drop_cross_test=int((~held).sum()),
                common_prepare_ms=elapsed, train_sha256=digest(X.indices),
                test_sha256=digest(Xt.indices), train_row_offsets=digest(X.indptr),
                test_row_offsets=digest(Xt.indptr), train_labels=digest(y), test_labels=digest(labels),
                vocabulary_sha256=digest(np.array(vectorizer.get_feature_names_out()[keep], dtype="U")))
    return X, y, Xt, labels, meta


def run(out, verify):
    files = [Path(__file__), ROOT / "experiments/document_vote_benchmark.py",
             ROOT / "src/minppi_mofs/document_vote_minpi.py", ROOT / "src/minppi_mofs/signal_vote_minpi.py",
             ROOT / "experiments/minpi_constant_transfer.py", ROOT / "experiments/compare_predictive_minpi.py",
             ROOT / "experiments/fast_minpi_five_source.py", ROOT / "docs/minpi-study/mail-vote-protocol.md"]
    state = dict(sources=ARCHIVES, methods=list(ARMS), seeds=list(SEEDS),
                 files={f.relative_to(ROOT).as_posix(): file_hash(f) for f in files})
    if verify:
        assert state == json.loads((out / "protocol.json").read_text())
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "protocol.json", state)
    X, y, Xt, labels, common = prepare()
    trials, audits, draws, cases = [], [], [], []
    for seed in SEEDS:
        meta = dict(common, seed=seed)
        times, audit, random = run_case(out, X, y, Xt, labels, meta, verify)
        trials.extend(times)
        audits.extend(audit)
        draws.extend(random)
        cases.append(meta)
        print(f"Completed seed {seed}", flush=True)
    panels, random = pd.DataFrame(cases), pd.DataFrame(draws)
    if verify:
        fields = panels.columns.drop("common_prepare_ms")
        pd.testing.assert_frame_equal(panels[fields], pd.read_csv(out / "panels.csv")[fields])
        pd.testing.assert_frame_equal(random, pd.read_csv(out / "random-draws.csv"), rtol=0, atol=1e-12)
        pd.DataFrame(audits).to_csv(out / "fresh-process-audit.csv", index=False)
    else:
        assert len(trials) == len(SEEDS) * len(ARMS) ** 2
        summarize(out, trials, panels, random)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.verify)
