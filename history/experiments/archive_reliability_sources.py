"""Snapshot exactly hashed experiment sources without changing active files."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]


def run(output):
    frozen = {}
    protocols = sorted((ROOT / "results/minpi-study").glob("reliability*/protocol.json"))
    assert protocols
    included = {}
    for file in protocols:
        state = json.loads(file.read_text(encoding="utf8"))
        # mail-a is a documented pre-evaluation parsing failure, not a result.
        if file.parent.name == "reliability-mail-20261117a":
            continue
        for key in ("sources", "source_sha256", "file_sha256"):
            for path, digest in state.get(key, {}).items():
                # Seed protocols call their named dataset checksums "sources";
                # their actual code paths are in file_sha256.
                if key == "sources" and "/" not in path and "\\" not in path:
                    continue
                if path in frozen:
                    assert frozen[path] == digest, path
                frozen[path] = digest
        included[file.relative_to(ROOT).as_posix()] = file.read_bytes()
    for path, digest in frozen.items():
        content = (ROOT / path).read_bytes()
        assert sha256(content).hexdigest() == digest, path
        included[path] = content
    for pattern in ("experiments/*reliability*.py", "experiments/verify_refined_document.py",
                    "experiments/audit_refined_document.py", "tests/test_*document*minpi.py",
                    "tests/test_next_vote_selector.py", "thesis-python.ps1",
                    "src/minppi_mofs/__init__.py"):
        for file in ROOT.glob(pattern):
            included[file.relative_to(ROOT).as_posix()] = file.read_bytes()
    # Public API integration followed the first median-three experiment.
    # Preserve its former API bytes only if their original digest matches.
    current = (ROOT / "src/minppi_mofs/support_selector.py").read_bytes()
    previous = current.replace(
        b"from minppi_mofs.reliability_document_minpi import reliability_document_top_from_training\n", b""
    ).replace(b', "reliability_vote"', b"").replace(
        b'                          "reliability_vote": reliability_document_top_from_training,\n', b""
    ).replace(
        b"    Reliability_vote uses one sketch and takes the minimum over its two\n"
        b"    cross-fit directions within each peer-count stratum.\n", b""
    )
    previous_state = json.loads((ROOT / "results/minpi-study/refined-document-20261117a/protocol.json").read_text())
    expected = previous_state["sources"]["src/minppi_mofs/support_selector.py"]
    old_api_available = sha256(previous).hexdigest() == expected
    if old_api_available:
        included["legacy-public-api/median-three-support_selector.py"] = previous
    manifest = dict(
        files={path: sha256(content).hexdigest() for path, content in included.items()},
        frozen_sources=len(frozen), median_three_public_api_restored=old_api_available,
        restore="Extract to an isolated thesis copy with the existing dependencies and locally archived data. "
                "Paths are relative to history. Do not extract over Master. Protocols record input hashes; "
                "this archive contains source and native libraries, not a second copy of all datasets. "
                "For the earlier median-three protocol use legacy-public-api/median-three-support_selector.py "
                "as src/minppi_mofs/support_selector.py in a separate copy if present.",
    )
    included["source-manifest.json"] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for path, content in sorted(included.items()):
            archive.writestr(path, content)
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == set(included)
        for path, content in included.items():
            assert archive.read(path) == content, path
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps(dict(files=len(included), frozen_sources=len(frozen), sha256=digest,
                          median_three_public_api_restored=old_api_available)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
