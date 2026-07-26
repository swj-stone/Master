# Coalition Cache and Nested Schedules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the content-validated coalition utility cache and deterministic nested random/guided estimator primitives required by grouped execution.

**Architecture:** A new `coalition_cache.py` module owns canonical cache identity, SQLite persistence, multi-utility evaluation, counters, and metric views. A new `nested_estimation.py` module owns budget-independent per-feature random streams, maximum-budget schedules, checkpoint evaluation, and hybrid derivation while the existing public estimators remain unchanged for legacy comparison.

**Tech Stack:** Python 3.13, NumPy, pandas, scikit-learn metrics, XGBoost, standard-library `sqlite3`, `hashlib`, canonical JSON, pytest.

## Global Constraints

- Do not change datasets, seeds, budgets `[10, 25, 50, 100, 200]`, XGBoost settings, utilities, or the configured backend.
- Do not add adaptive CPU/GPU routing or concurrent GPU workers.
- Cache classification utilities from one `predict_proba` call and preserve a prediction SHA-256 digest.
- Keep validation and test cache identities structurally distinct.
- Cache corruption and identity conflicts fail closed.
- Retain current `min_ppi.py` public APIs as the legacy benchmark path.
- Use the project interpreter `C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\venv\Scripts\python.exe`.
- Do not stage or commit Git changes; the user owns Git.
- Follow red-green-refactor and stop at the review checkpoint after every task.

## File map

- Create `src/minppi_mofs/coalition_cache.py`: cache identity, canonical serialization, SQLite store, multi-utility evaluator, counters, and metric-specific views.
- Create `src/minppi_mofs/nested_estimation.py`: immutable random/guided master schedules, digests, checkpoint evaluators, and hybrid derivation.
- Create `tests/test_coalition_cache.py`: cache identity, persistence, corruption, multi-utility, namespace, and counter tests.
- Create `tests/test_nested_estimation.py`: prefix, determinism, checkpoint, early-stop, and hybrid tests.
- Modify `tests/test_min_ppi.py`: lock the existing public estimator behavior as the legacy path.

---

### Task 1: Canonical cache identity and payload types

**Files:**
- Create: `src/minppi_mofs/coalition_cache.py`
- Create: `tests/test_coalition_cache.py`

**Interfaces:**
- Produces: `CACHE_SCHEMA_VERSION: str`
- Produces: `CoalitionCacheIdentity` with `canonical_json()` and `digest`
- Produces: `CoalitionUtilityRecord`
- Produces: `canonical_coalition(feature_order, coalition) -> tuple[str, ...]`
- Produces: `hash_numeric_frame(frame) -> str` and
  `hash_numeric_array(values) -> str`
- Consumes later: SQLite store and grouped stage adapters use these exact types.

- [ ] **Step 1: Write failing identity tests**

```python
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from minppi_mofs.coalition_cache import (
    CACHE_SCHEMA_VERSION,
    CoalitionCacheIdentity,
    canonical_coalition,
    hash_numeric_array,
    hash_numeric_frame,
)


def identity(**changes):
    value = CoalitionCacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        protocol_hash="0" * 64,
        dataset_id="toy",
        seed=42,
        namespace="selection_validation",
        train_features_sha256="a" * 64,
        train_target_sha256="b" * 64,
        eval_features_sha256="c" * 64,
        eval_target_sha256="d" * 64,
        ordered_features=("a", "b", "c"),
        task="binary",
        class_labels=(0, 1),
        model_params={
            "learning_rate": 0.1,
            "max_depth": 4,
            "n_estimators": 100,
            "n_jobs": 1,
        },
        backend="cuda",
        model_seed=42,
        library_versions={"numpy": "2", "xgboost": "3"},
    )
    return replace(value, **changes)


def test_identity_is_canonical_and_field_sensitive():
    first = identity()
    assert first.canonical_json() == identity().canonical_json()
    assert first.digest == identity().digest
    for field, value in {
        "namespace": "final_test",
        "eval_features_sha256": "e" * 64,
        "backend": "cpu",
        "model_seed": 123,
    }.items():
        assert replace(first, **{field: value}).digest != first.digest


def test_canonical_coalition_uses_frozen_feature_order():
    assert canonical_coalition(("b", "a", "c"), {"a", "b"}) == ("b", "a")
    with pytest.raises(ValueError, match="outside"):
        canonical_coalition(("a",), {"unknown"})


def test_numeric_hashes_bind_order_shape_dtype_and_values():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert hash_numeric_frame(frame) == hash_numeric_frame(frame.copy())
    assert hash_numeric_frame(frame[["b", "a"]]) != hash_numeric_frame(frame)
    assert hash_numeric_array(np.array([0, 1])) != hash_numeric_array(
        np.array([1, 0])
    )
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```powershell
Set-Location 'C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master'
& '..\venv\Scripts\python.exe' -m pytest tests/test_coalition_cache.py -q -p no:cacheprovider
```

Expected: collection fails because `minppi_mofs.coalition_cache` does not exist.

- [ ] **Step 3: Implement immutable validated types**

Create:

```python
CACHE_SCHEMA_VERSION = "coalition_utility_v1"
_NAMESPACES = frozenset({"selection_validation", "baseline_internal", "final_test"})
_TASKS = frozenset({"binary", "multiclass", "regression"})
_BACKENDS = frozenset({"cpu", "cuda"})


@dataclass(frozen=True)
class CoalitionCacheIdentity:
    schema_version: str
    protocol_hash: str
    dataset_id: str
    seed: int
    namespace: str
    train_features_sha256: str
    train_target_sha256: str
    eval_features_sha256: str
    eval_target_sha256: str
    ordered_features: tuple[str, ...]
    task: str
    class_labels: tuple[int, ...]
    model_params: Mapping[str, object]
    backend: str
    model_seed: int
    library_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError("cache identity schema is invalid")
        if self.namespace not in _NAMESPACES:
            raise ValueError("cache identity namespace is invalid")
        if self.task not in _TASKS:
            raise ValueError("cache identity task is invalid")
        if self.backend not in _BACKENDS:
            raise ValueError("cache identity backend is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.seed, self.model_seed)
        ):
            raise ValueError("cache identity seeds must be non-negative integers")
        if (
            not self.ordered_features
            or len(set(self.ordered_features)) != len(self.ordered_features)
            or not all(isinstance(item, str) and item for item in self.ordered_features)
        ):
            raise ValueError("cache identity features must be non-empty unique strings")
        hash_fields = (
            self.protocol_hash,
            self.train_features_sha256,
            self.train_target_sha256,
            self.eval_features_sha256,
            self.eval_target_sha256,
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hash_fields
        ):
            raise ValueError("cache identity hashes must be lowercase SHA-256")
        if self.task == "regression" and self.class_labels:
            raise ValueError("regression cache identity cannot have class labels")
        if self.task == "binary" and self.class_labels != (0, 1):
            raise ValueError("binary class labels must be zero-based and contiguous")
        if self.task == "multiclass" and self.class_labels != tuple(
            range(len(self.class_labels))
        ):
            raise ValueError("multiclass labels must be zero-based and contiguous")
        object.__setattr__(
            self,
            "model_params",
            MappingProxyType(dict(sorted(self.model_params.items()))),
        )
        object.__setattr__(
            self,
            "library_versions",
            MappingProxyType(dict(sorted(self.library_versions.items()))),
        )

    def canonical_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "protocol_hash": self.protocol_hash,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "namespace": self.namespace,
            "train_features_sha256": self.train_features_sha256,
            "train_target_sha256": self.train_target_sha256,
            "eval_features_sha256": self.eval_features_sha256,
            "eval_target_sha256": self.eval_target_sha256,
            "ordered_features": list(self.ordered_features),
            "task": self.task,
            "class_labels": list(self.class_labels),
            "model_params": dict(self.model_params),
            "backend": self.backend,
            "model_seed": self.model_seed,
            "library_versions": dict(self.library_versions),
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CoalitionUtilityRecord:
    coalition: tuple[str, ...]
    utilities: Mapping[str, float]
    prediction_shape: tuple[int, ...]
    prediction_dtype: str
    prediction_sha256: str
    fitted: bool


def canonical_coalition(
    feature_order: Sequence[str], coalition: AbstractSet[str]
) -> tuple[str, ...]:
    ordered = tuple(feature_order)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("feature order must be non-empty and unique")
    if not isinstance(coalition, AbstractSet):
        raise TypeError("coalition must be a set-like value")
    unknown = set(coalition) - set(ordered)
    if unknown:
        raise ValueError(f"coalition contains features outside authority: {sorted(unknown)}")
    return tuple(feature for feature in ordered if feature in coalition)


def _hash_numeric_payload(
    values: np.ndarray, *, labels: Sequence[str] = ()
) -> str:
    contiguous = np.ascontiguousarray(values)
    metadata = json.dumps(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "labels": list(labels),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(metadata + b"\0" + contiguous.tobytes()).hexdigest()


def hash_numeric_frame(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("frame must have unique columns")
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(values.astype(float)).all():
        raise ValueError("frame must be finite")
    return _hash_numeric_payload(values, labels=tuple(map(str, frame.columns)))


def hash_numeric_array(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.ndim != 1 or not np.isfinite(array.astype(float)).all():
        raise ValueError("array must be finite and one-dimensional")
    return _hash_numeric_payload(array)
```

Import `MappingProxyType` from `types` and `AbstractSet`, `Mapping`, and
`Sequence` from `collections.abc`. Reject non-string model/version keys and
non-finite model numeric values before freezing the mappings.

- [ ] **Step 4: Add invalid-schema and invalid-class tests**

Add tests that reject:

```python
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "unknown", "schema"),
        ("namespace", "test", "namespace"),
        ("backend", "auto", "backend"),
        ("ordered_features", ("a", "a"), "unique"),
        ("class_labels", (1, 2), "zero-based"),
    ],
)
def test_identity_rejects_invalid_authority(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        identity(**{field: value})
```

For regression, require `class_labels == ()`; for classification require
zero-based contiguous labels and exactly two labels for `binary`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_coalition_cache.py -q -p no:cacheprovider
```

Expected: all identity and canonical-coalition tests pass.

- [ ] **Step 6: Review checkpoint**

Report the new public names, focused test count, and changed files. Do not run a
Git write command.

---

### Task 2: Transactional SQLite cache and integrity failures

**Files:**
- Modify: `src/minppi_mofs/coalition_cache.py`
- Modify: `tests/test_coalition_cache.py`

**Interfaces:**
- Consumes: `CoalitionCacheIdentity`, `CoalitionUtilityRecord`
- Produces: `SQLiteCoalitionStore(path, identity, commit_every_fits)`
- Produces: `get(coalition)`, `put(record)`, `flush()`, `close()`
- Produces properties: `entry_count`, `pending_write_count`

- [ ] **Step 1: Write persistence and identity-conflict tests**

```python
def test_sqlite_store_persists_one_canonical_record(tmp_path):
    path = tmp_path / "cache.sqlite3"
    record = CoalitionUtilityRecord(
        coalition=("a",),
        utilities={"accuracy": 0.75, "roc_auc": 0.8, "pr_auc": 0.7},
        prediction_shape=(4, 2),
        prediction_dtype="float64",
        prediction_sha256="f" * 64,
        fitted=True,
    )
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as store:
        assert store.get(("a",)) is None
        store.put(record)
        assert store.get(("a",)) == record
    with SQLiteCoalitionStore(path, identity(), commit_every_fits=1) as reopened:
        assert reopened.get(("a",)) == record


def test_sqlite_store_rejects_foreign_identity(tmp_path):
    path = tmp_path / "cache.sqlite3"
    SQLiteCoalitionStore(path, identity(), commit_every_fits=1).close()
    with pytest.raises(ValueError, match="identity"):
        SQLiteCoalitionStore(
            path, identity(eval_features_sha256="9" * 64), commit_every_fits=1
        )
```

- [ ] **Step 2: Run the two tests and confirm missing store failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_coalition_cache.py -q -p no:cacheprovider -k 'sqlite_store'
```

Expected: failure because `SQLiteCoalitionStore` is undefined.

- [ ] **Step 3: Implement schema creation and canonical rows**

Use this schema:

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE coalition_utility (
    coalition_key TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
);
```

Implementation requirements:

```python
class SQLiteCoalitionStore:
    def __init__(
        self,
        path: Path,
        identity: CoalitionCacheIdentity,
        *,
        commit_every_fits: int,
    ) -> None:
        if isinstance(commit_every_fits, bool) or not isinstance(
            commit_every_fits, int
        ) or commit_every_fits < 1:
            raise ValueError("commit_every_fits must be a positive integer")
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("cache path must not be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._commit_every_fits = commit_every_fits
        self._pending_write_count = 0
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(SCHEMA_SQL)
        rows = dict(self._connection.execute("SELECT key, value FROM metadata"))
        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "identity_json": identity.canonical_json(),
            "identity_sha256": identity.digest,
        }
        if not rows:
            self._connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                tuple(expected.items()),
            )
            self._connection.commit()
        elif rows != expected:
            self._connection.close()
            raise ValueError("cache metadata conflicts with identity")

    def get(self, coalition: tuple[str, ...]) -> CoalitionUtilityRecord | None:
        coalition_key = json.dumps(
            list(coalition), ensure_ascii=False, separators=(",", ":")
        )
        row = self._connection.execute(
            "SELECT payload_json, payload_sha256 FROM coalition_utility "
            "WHERE coalition_key = ?",
            (coalition_key,),
        ).fetchone()
        if row is None:
            return None
        payload_json, expected_sha256 = row
        actual_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("coalition cache payload checksum is invalid")
        payload = json.loads(payload_json)
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical != payload_json:
            raise ValueError("coalition cache payload is not canonical")
        expected_keys = {
            "coalition",
            "utilities",
            "prediction_shape",
            "prediction_dtype",
            "prediction_sha256",
            "fitted",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("coalition cache payload schema is invalid")
        record = CoalitionUtilityRecord(
            coalition=tuple(payload["coalition"]),
            utilities=MappingProxyType(dict(payload["utilities"])),
            prediction_shape=tuple(payload["prediction_shape"]),
            prediction_dtype=payload["prediction_dtype"],
            prediction_sha256=payload["prediction_sha256"],
            fitted=payload["fitted"],
        )
        if record.coalition != coalition:
            raise ValueError("coalition cache key conflicts with payload")
        return record

    def put(self, record: CoalitionUtilityRecord) -> None:
        coalition_key = json.dumps(
            list(record.coalition), ensure_ascii=False, separators=(",", ":")
        )
        payload = {
            "coalition": list(record.coalition),
            "utilities": dict(record.utilities),
            "prediction_shape": list(record.prediction_shape),
            "prediction_dtype": record.prediction_dtype,
            "prediction_sha256": record.prediction_sha256,
            "fitted": record.fitted,
        }
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        existing = self.get(record.coalition)
        if existing is not None and existing != record:
            raise ValueError("coalition cache write conflicts with existing row")
        if existing is None:
            self._connection.execute(
                "INSERT INTO coalition_utility"
                "(coalition_key, payload_json, payload_sha256) VALUES (?, ?, ?)",
                (coalition_key, payload_json, payload_sha256),
            )
            self._pending_write_count += int(record.fitted)
            if self._pending_write_count >= self._commit_every_fits:
                self.flush()

    def flush(self) -> None:
        self._connection.commit()
        self._pending_write_count = 0

    def close(self) -> None:
        self.flush()
        self._connection.close()
```

Define `SCHEMA_SQL` as the exact two `CREATE TABLE IF NOT EXISTS` statements
shown above. Add `__enter__`, `__exit__`, `entry_count`, and
`pending_write_count` as direct wrappers over the connection and stored
counter. Validate every reconstructed record field before returning it.

Store `schema_version`, `identity_json`, and `identity_sha256` in `metadata`.
Use `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=FULL`, parameterized SQL,
canonical JSON with `allow_nan=False`, and SHA-256 over exact payload bytes.
Reject symlink cache paths and nonpositive/bool commit intervals.

- [ ] **Step 4: Add corruption and batch-commit tests**

Use direct `sqlite3.connect(path)` only in the test to mutate
`payload_sha256`, `payload_json`, and metadata one at a time. Assert reopening
or reading raises `ValueError` containing `checksum`, `canonical`, or
`metadata`. Add a transaction test that writes 49 fitted rows with
`commit_every_fits=50`, calls `flush()`, reopens, and reads all 49.

- [ ] **Step 5: Run cache tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_coalition_cache.py -q -p no:cacheprovider
```

Expected: all tests pass; no `-wal` file is required after clean close.

- [ ] **Step 6: Review checkpoint**

Report schema version, transaction policy, corruption cases exercised, and
test results. Do not stage or commit.

---

### Task 3: One-fit multi-utility evaluator and cache views

**Files:**
- Modify: `src/minppi_mofs/coalition_cache.py`
- Modify: `tests/test_coalition_cache.py`
- Test compatibility: `tests/test_utility.py`

**Interfaces:**
- Produces: `CoalitionCacheStats`
- Produces: `CoalitionUtilityCache(identity, X_train, y_train, X_eval, y_eval, store)`
- Produces: `CoalitionUtilityView(cache, utility, alpha)`
- `CoalitionUtilityCache.evaluate(coalition, utility) -> float`
- `CoalitionUtilityView` provides `evaluate`, `set_threshold`, `is_winning`,
  `eval_marginal`, `model_fit_count`, and `coalition_count`.

- [ ] **Step 1: Write the one-fit/two-utility test**

```python
def test_multiclass_metrics_share_one_prediction(monkeypatch, tmp_path):
    calls = []
    predictions = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
    )

    def fake_fit_predict(*args, **kwargs):
        calls.append(kwargs)
        return predictions.copy()

    monkeypatch.setattr("minppi_mofs.coalition_cache.fit_predict", fake_fit_predict)
    cache = make_multiclass_cache(tmp_path)
    accuracy = cache.evaluate({"a"}, "accuracy")
    auc = cache.evaluate({"a"}, "ovr_auc")
    record = cache.record({"a"})

    assert len(calls) == 1
    assert accuracy == compute_utility(Y, predictions, metric="accuracy")
    assert auc == compute_utility(Y, predictions, metric="ovr_auc")
    assert record.prediction_sha256 == prediction_digest(predictions)
    assert cache.stats.physical_model_fit_count == 1
    assert cache.stats.cache_hit_count == 2
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_coalition_cache.py::test_multiclass_metrics_share_one_prediction -q -p no:cacheprovider
```

Expected: failure because `CoalitionUtilityCache` is undefined.

- [ ] **Step 3: Implement task-specific metric materialization**

Use:

```python
_TASK_METRICS = {
    "binary": ("accuracy", "pr_auc", "roc_auc"),
    "multiclass": ("accuracy", "ovr_auc"),
    "regression": ("r2", "rmse"),
}


def prediction_digest(predictions: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(predictions)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()
```

`CoalitionUtilityCache.record()` must:

1. canonicalize the coalition;
2. return a hot/store hit when present;
3. produce deterministic empty-subset predictions without a fit;
4. otherwise call `modeling.fit_predict` exactly once;
5. compute every metric in `_TASK_METRICS[task]`;
6. persist one `CoalitionUtilityRecord`; and
7. update physical-fit and hit counters.

The constructor receives already validated frames/targets and a
`SQLiteCoalitionStore`; it must verify frame column order equals the identity.
Expose this immutable snapshot:

```python
@dataclass(frozen=True)
class CoalitionCacheStats:
    physical_model_fit_count: int
    cache_hit_count: int
    coalition_count: int
```

`CoalitionUtilityCache.stats` returns current-process physical fits and cache
hits plus the persistent store entry count. It starts physical fits and hits at
zero after reopening.

- [ ] **Step 4: Implement metric-specific views**

```python
class CoalitionUtilityView:
    def __init__(
        self, cache: CoalitionUtilityCache, *, utility: str, alpha: float
    ) -> None:
        self._cache = cache
        self._utility = utility
        self._seen: set[tuple[str, ...]] = set()
        self.set_threshold(alpha)

    def set_threshold(self, alpha: float) -> None:
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or not 0.0 < float(alpha) <= 1.0
        ):
            raise ValueError("alpha must be in (0, 1]")
        self.acc_full = self.evaluate(set(self._cache.feature_order))
        self.threshold = self.acc_full * float(alpha)

    def evaluate(self, coalition: set[str]) -> float:
        self._seen.add(
            canonical_coalition(self._cache.feature_order, coalition)
        )
        return self._cache.evaluate(coalition, self._utility)

    def is_winning(self, coalition: set[str]) -> bool:
        return self.evaluate(coalition) >= self.threshold

    def eval_marginal(self, coalition: set[str], feature: str) -> int:
        return int(self.is_winning(coalition | {feature})) - int(
            self.is_winning(coalition)
        )

    @property
    def model_fit_count(self) -> int:
        return sum(bool(key) for key in self._seen)

    @property
    def coalition_count(self) -> int:
        return len(self._seen)
```

Expose deterministic logical unique-coalition accounting separately from
physical store counters. Do not make `model_fit_count` depend on whether a
different view executed first.

- [ ] **Step 5: Add binary, regression, empty, reopen, and namespace tests**

Assert:

- binary materializes exactly `accuracy`, `pr_auc`, and `roc_auc`;
- regression materializes exactly `r2` and `rmse`;
- empty coalition has `fitted is False`;
- reopening returns identical utilities/digest without a fit;
- a `final_test` identity cannot open a `selection_validation` database; and
- cached/uncached values agree within `1e-12`.

- [ ] **Step 6: Run focused and compatibility tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_coalition_cache.py tests/test_utility.py -q -p no:cacheprovider
```

Expected: both files pass.

- [ ] **Step 7: Review checkpoint**

Report fits, hits, task metrics, digest behavior, and compatibility results. Do
not stage or commit.

---

### Task 4: Nested Random Monte Carlo schedule and checkpoints

**Files:**
- Create: `src/minppi_mofs/nested_estimation.py`
- Create: `tests/test_nested_estimation.py`
- Modify: `tests/test_min_ppi.py`

**Interfaces:**
- Produces: `NESTED_SCHEDULE_SCHEMA_VERSION`
- Produces: `NestedRandomSchedule`
- Produces: `make_nested_random_schedule(features, budgets, seed)`
- Produces: `nested_random_schedule_digest(schedule)`
- Produces: `estimate_random_checkpoints(schedule, view)`

- [ ] **Step 1: Write deterministic-prefix tests**

```python
def test_random_schedule_is_budget_independent_and_prefix_nested():
    small = make_nested_random_schedule(FEATURES, budgets=(10, 25), seed=42)
    large = make_nested_random_schedule(
        FEATURES, budgets=(10, 25, 50, 100, 200), seed=42
    )
    assert small.trials[0] == large.trials[0][:25]
    assert small.trials[1] == large.trials[1][:25]
    assert small.checkpoints == (10, 25)
    assert large.checkpoints == (10, 25, 50, 100, 200)
    assert nested_random_schedule_digest(large) == nested_random_schedule_digest(
        make_nested_random_schedule(
            FEATURES, budgets=(10, 25, 50, 100, 200), seed=42
        )
    )
```

Add a test proving feature 1's first 25 coalitions are unchanged when feature
0's requested maximum changes. Add different-seed and feature-order digest
tests.

- [ ] **Step 2: Run and confirm missing nested module failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_nested_estimation.py -q -p no:cacheprovider
```

Expected: collection fails because `nested_estimation.py` does not exist.

- [ ] **Step 3: Implement immutable per-feature schedule generation**

Define:

```python
NESTED_SCHEDULE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NestedRandomSchedule:
    features: tuple[str, ...]
    seed: int
    checkpoints: tuple[int, ...]
    trials: tuple[tuple[frozenset[str], ...], ...]


@dataclass(frozen=True)
class NestedProgress:
    algorithm: str
    feature_index: int
    feature_count: int
    budget: int
    max_budget: int


def _feature_rng(seed: int, algorithm_id: int, feature_index: int):
    return np.random.default_rng(
        np.random.SeedSequence(
            [seed, algorithm_id, feature_index, NESTED_SCHEDULE_SCHEMA_VERSION]
        )
    )
```

For each feature, draw `min(max(checkpoints), 2 ** (n - 1))` coalitions from
the other features using that feature's RNG. Validate strictly increasing
positive unique checkpoints and immutable unique feature order.

- [ ] **Step 4: Write checkpoint evaluator tests**

Use a recording view whose `eval_marginal` returns a predefined sequence.
Assert:

```python
result = estimate_random_checkpoints(schedule, view)
assert tuple(result) == (10, 25, 50, 100, 200)
assert result[10]["a"] == expected_after_10
assert result[25]["a"] == expected_after_25
assert view.calls_for("a") <= 200
```

Add an early-stop test where trial 3 returns `-1`; all later checkpoints must be
`-1` and no fourth physical marginal is requested for that feature.

- [ ] **Step 5: Implement checkpoint evaluation**

```python
def estimate_random_checkpoints(
    schedule: NestedRandomSchedule,
    view: MarginalUtilityView,
    *,
    progress: Callable[[NestedProgress], None] | None = None,
) -> dict[int, dict[str, int]]:
    checkpoints = schedule.checkpoints
    scores_at = {budget: {} for budget in checkpoints}
    for feature_index, feature in enumerate(schedule.features):
        best = 1
        trials = schedule.trials[feature_index]
        cursor = 0
        for budget in checkpoints:
            target = min(budget, len(trials))
            while cursor < target and best != -1:
                best = min(best, view.eval_marginal(set(trials[cursor]), feature))
                cursor += 1
            scores_at[budget][feature] = best
            if progress is not None:
                progress(
                    NestedProgress(
                        algorithm="random",
                        feature_index=feature_index + 1,
                        feature_count=len(schedule.features),
                        budget=budget,
                        max_budget=checkpoints[-1],
                    )
                )
    return scores_at
```

Record scores at every checkpoint. Emit progress after a feature boundary and
never consume RNG during evaluation. Validate that the view's feature order
matches the schedule.

- [ ] **Step 6: Lock legacy public behavior**

Add a test showing
`estimate_minppi_random_mc(X_train, X_val, y_train, y_val, K=25,
acc_cache=recording_cache, seed=42)` retains its existing output and does not
import or call the nested implementation. This keeps a valid legacy comparison
path.

- [ ] **Step 7: Run tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_nested_estimation.py tests/test_min_ppi.py -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 8: Review checkpoint**

Report schedule digests, prefix cases, early-stop case, and tests. Do not stage
or commit.

---

### Task 5: Nested guided schedule and checkpoints

**Files:**
- Modify: `src/minppi_mofs/nested_estimation.py`
- Modify: `tests/test_nested_estimation.py`
- Modify: `tests/test_min_ppi.py`

**Interfaces:**
- Produces: `NestedGuidedSchedule`
- Produces: `make_nested_guided_schedule(features, budgets, seed)`
- Produces: `nested_guided_schedule_digest(schedule)`
- Produces: `estimate_guided_checkpoints(schedule, view, mi_scores, correlation_neighbors)`
- Keeps: `make_algorithm_b_perturbation_schedule` as the legacy generator.

- [ ] **Step 1: Write guided-prefix and legacy-separation tests**

```python
def test_guided_schedule_uses_independent_feature_streams():
    schedule = make_nested_guided_schedule(
        FEATURES, budgets=(10, 25, 50, 100, 200), seed=42
    )
    prefix = make_nested_guided_schedule(FEATURES, budgets=(10, 25), seed=42)
    for index in range(len(FEATURES)):
        assert prefix.seed_orders[index] == schedule.seed_orders[index]
        assert prefix.trials[index] == schedule.trials[index][:25]


def test_legacy_algorithm_b_schedule_remains_available():
    legacy = make_algorithm_b_perturbation_schedule(FEATURES, K_adv=25, seed=42)
    nested = make_nested_guided_schedule(FEATURES, budgets=(25,), seed=42)
    assert legacy.K_adv == nested.max_budget == 25
```

Do not assert the old and new schedules are equal; they deliberately use
different stream structure.

- [ ] **Step 2: Run and confirm failures**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_nested_estimation.py -q -p no:cacheprovider -k guided
```

Expected: failures for missing guided names.

- [ ] **Step 3: Implement guided master schedule**

Reuse the immutable `AlgorithmBPerturbationTrial` representation from
`min_ppi.py`. Generate one `seed_order` and `max(budgets)` trials per feature
from `_feature_rng(seed, GUIDED_ALGORITHM_ID, feature_index)`. Store checkpoint
budgets in the new schedule and include all trials in its digest.

Do not modify the legacy generator's random-stream behavior.

- [ ] **Step 4: Implement guided checkpoint evaluation**

Extract the deterministic seed-coalition construction into a pure helper:

```python
def guided_seed_coalition(
    feature: str,
    feature_order: tuple[str, ...],
    seed_order: tuple[str, ...],
    *,
    mi_scores: Mapping[str, float] | None,
    correlation_neighbors: Mapping[str, Sequence[str]] | None,
) -> frozenset[str]:
    others = tuple(item for item in feature_order if item != feature)
    if mi_scores is None:
        top_half = feature_order
    else:
        rank = {item: index for index, item in enumerate(feature_order)}
        top_half = tuple(
            item
            for item, _ in sorted(
                ((item, mi_scores[item]) for item in feature_order),
                key=lambda pair: (-pair[1], rank[pair[0]]),
            )[: max(1, len(feature_order) // 2)]
        )
    correlated = (
        tuple(
            item
            for item in correlation_neighbors.get(feature, ())
            if item in others
        )
        if correlation_neighbors is not None
        else ()
    )
    if correlated:
        selected = set(correlated[:10])
        for item in top_half:
            if len(selected) >= min(20, len(feature_order) // 3):
                break
            if item != feature:
                selected.add(item)
        return frozenset(selected)
    seed_size = min(20, len(feature_order) // 3)
    top_pool = set(top_half)
    candidates = [
        item for item in seed_order if item in top_pool and item in others
    ]
    return frozenset(candidates[:seed_size])
```

Then implement:

```python
def estimate_guided_checkpoints(
    schedule: NestedGuidedSchedule,
    view: MarginalUtilityView,
    *,
    mi_scores: Mapping[str, float] | None,
    correlation_neighbors: Mapping[str, Sequence[str]] | None,
    progress: Callable[[NestedProgress], None] | None = None,
) -> dict[int, dict[str, int]]:
    scores_at = {budget: {} for budget in schedule.checkpoints}
    for feature_index, feature in enumerate(schedule.features):
        seed_coalition = guided_seed_coalition(
            feature,
            schedule.features,
            schedule.seed_orders[feature_index],
            mi_scores=mi_scores,
            correlation_neighbors=correlation_neighbors,
        )
        best = 1
        trials = schedule.trials[feature_index]
        cursor = 0
        for budget in schedule.checkpoints:
            while cursor < min(budget, len(trials)) and best != -1:
                trial = trials[cursor]
                coalition = set(seed_coalition)
                for candidate, draw in zip(
                    schedule.features, trial.removal_draws, strict=True
                ):
                    if candidate in coalition and draw < 0.3:
                        coalition.discard(candidate)
                outside = [
                    item
                    for item in trial.add_order
                    if item not in coalition and item != feature
                ]
                coalition.update(outside[: min(trial.add_count, len(outside))])
                best = min(best, view.eval_marginal(coalition, feature))
                cursor += 1
            scores_at[budget][feature] = best
            if progress is not None:
                progress(
                    NestedProgress(
                        algorithm="guided",
                        feature_index=feature_index + 1,
                        feature_count=len(schedule.features),
                        budget=budget,
                        max_budget=schedule.checkpoints[-1],
                    )
                )
    return scores_at
```

Apply each stored perturbation trial exactly as the legacy guided estimator
does. Record checkpoint scores and propagate an early `-1`.

- [ ] **Step 5: Add direct-reference tests**

For a small deterministic schedule, compare each checkpoint with a simple test
reference loop that applies the same trials. Cover guidance combinations:

```python
[
    (None, None),
    (MI, None),
    (None, NEIGHBORS),
    (MI, NEIGHBORS),
]
```

Assert schedule evaluation does not mutate either guidance mapping.

- [ ] **Step 6: Run nested and legacy tests**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_nested_estimation.py tests/test_min_ppi.py -q -p no:cacheprovider
```

Expected: all tests pass and legacy schedule tests remain unchanged.

- [ ] **Step 7: Review checkpoint**

Report the guided digest cases, four guidance modes, and legacy compatibility.
Do not stage or commit.

---

### Task 6: Exact reuse and hybrid checkpoint derivation

**Files:**
- Modify: `src/minppi_mofs/nested_estimation.py`
- Modify: `tests/test_nested_estimation.py`

**Interfaces:**
- Produces: `derive_hybrid_checkpoints(feature_count, checkpoints, exact_scores, random_scores, guided_scores)`
- Produces: `materialize_exact_checkpoints(checkpoints, exact_scores)`

- [ ] **Step 1: Write dispatcher-boundary tests**

```python
@pytest.mark.parametrize(
    ("feature_count", "expected"),
    [
        (20, EXACT),
        (21, GUIDED),
        (75, GUIDED),
        (76, {"a": min(RANDOM["a"], GUIDED["a"])}),
    ],
)
def test_hybrid_derivation_matches_dispatcher_boundary(feature_count, expected):
    result = derive_hybrid_checkpoints(
        feature_count=feature_count,
        checkpoints=(10,),
        exact_scores=EXACT,
        random_scores={10: RANDOM},
        guided_scores={10: GUIDED},
    )
    assert result[10] == expected
```

Add tests that reject missing checkpoint keys, inconsistent feature order,
scores outside `{-1, 0, 1}`, exact absent for `n <= 20`, and random absent for
`n > 75`.

- [ ] **Step 2: Run and confirm missing derivation failure**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_nested_estimation.py -q -p no:cacheprovider -k 'hybrid or exact'
```

Expected: failures for undefined derivation helpers.

- [ ] **Step 3: Implement exact materialization and hybrid derivation**

```python
def materialize_exact_checkpoints(
    checkpoints: Sequence[int], exact_scores: Mapping[str, int]
) -> dict[int, dict[str, int]]:
    validated = _validated_scores(exact_scores)
    return {budget: dict(validated) for budget in _validated_budgets(checkpoints)}


def derive_hybrid_checkpoints(
    *,
    feature_count: int,
    checkpoints: Sequence[int],
    exact_scores: Mapping[str, int] | None,
    random_scores: Mapping[int, Mapping[str, int]] | None,
    guided_scores: Mapping[int, Mapping[str, int]] | None,
) -> dict[int, dict[str, int]]:
    budgets = _validated_budgets(checkpoints)
    if feature_count <= 20:
        if exact_scores is None:
            raise ValueError("hybrid exact component is required")
        return materialize_exact_checkpoints(budgets, exact_scores)
    if guided_scores is None or set(guided_scores) != set(budgets):
        raise ValueError("hybrid guided checkpoints are incomplete")
    guided = {
        budget: _validated_scores(guided_scores[budget]) for budget in budgets
    }
    if feature_count <= 75:
        return {budget: dict(guided[budget]) for budget in budgets}
    if random_scores is None or set(random_scores) != set(budgets):
        raise ValueError("hybrid random checkpoints are incomplete")
    random_values = {
        budget: _validated_scores(random_scores[budget]) for budget in budgets
    }
    if any(
        tuple(random_values[budget]) != tuple(guided[budget])
        for budget in budgets
    ):
        raise ValueError("hybrid component feature orders conflict")
    return {
        budget: {
            feature: min(random_values[budget][feature], guided[budget][feature])
            for feature in random_values[budget]
        }
        for budget in budgets
    }
```

Return new dictionaries so callers cannot mutate component results.

- [ ] **Step 4: Add a direct-hybrid equivalence test**

On a toy frame and one fixed nested schedule:

1. evaluate random and guided checkpoints through a recording utility view;
2. derive hybrid;
3. run a direct reference that invokes the same component loops and takes their
   documented dispatcher result; and
4. assert exact equality at every checkpoint.

- [ ] **Step 5: Run core plan verification**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests/test_coalition_cache.py tests/test_nested_estimation.py tests/test_min_ppi.py tests/test_utility.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 6: Run the full suite**

Run:

```powershell
& '..\venv\Scripts\python.exe' -m pytest tests -q -p no:cacheprovider
```

Expected: all existing and new tests pass, with only the repository's
previously accepted skip.

- [ ] **Step 7: Review checkpoint**

Report the exact/hybrid boundary evidence, focused and full-suite totals, and
all changed files. Do not stage or commit.
