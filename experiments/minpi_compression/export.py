"""Bundle the sealed prediction arrays and inner selection traces for independent scoring."""
import argparse
from zipfile import ZIP_DEFLATED, ZipFile
import json

from experiments.minpi_compression.benchmark import DESIGN, EVIDENCE, RUN, digest


def export(phase):
    folder = RUN / phase
    lock = json.loads((folder/"lock.json").read_bytes())
    expected = {f"{name}-{seed}" for name in DESIGN[phase] for seed in DESIGN["seeds"]}
    assert set(lock) == expected
    target = EVIDENCE/phase/"forecasts.zip"
    target.parent.mkdir(exist_ok=True,parents=True)
    stage = target.with_suffix(".partial")
    with ZipFile(stage, "w", compression=ZIP_DEFLATED, compresslevel=6) as out:
        out.write(folder/"lock.json","lock.json")
        for key in sorted(expected):
            path = folder/key
            assert digest(path/"selection.json") == lock[key]
            assert digest(path/"predictions.npz") == json.loads((path/"selection.json").read_bytes())["predictions"]
            for name in ["selection.json","predictions.npz"]:
                out.write(path/name,f"{key}/{name}")
    with ZipFile(stage) as out:
        assert out.testzip() is None and len(out.namelist()) == 1 + len(expected)*2
    stage.replace(target)
    print(phase, target.stat().st_size, digest(target))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["pilot","confirmation"])
    export(p.parse_args().phase)
