"""Regression checks for the project's install metadata."""

from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]

RUNTIME_DEPENDENCIES = [
    "matplotlib==3.11.0",
    "networkx==3.6.1",
    "numba==0.66.0",
    "numpy==2.4.6",
    "pandas==3.0.3",
    "pyyaml==6.0.3",
    "scikit-learn==1.9.0",
    "scipy==1.18.0",
    "shap==0.52.0",
    "statsmodels==0.14.6",
    "tqdm==4.68.3",
    "xgboost==3.3.0",
]

DEV_DEPENDENCIES = ["pytest==8.4.2", "ruff==0.15.15"]


def _project_metadata() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as metadata_file:
        return tomllib.load(metadata_file)["project"]


def _dependency_name(requirement: str) -> str:
    return requirement.split("==", maxsplit=1)[0].lower().replace("_", "-")


class TestPackagingMetadata(unittest.TestCase):
    def test_project_metadata_has_pinned_python_312_runtime_dependencies(self) -> None:
        """A runtime dependency moved out of project metadata breaks installation."""
        project = _project_metadata()
        runtime_dependencies = project.get("dependencies")

        self.assertEqual(project["requires-python"], ">=3.12")
        self.assertEqual(runtime_dependencies, RUNTIME_DEPENDENCIES)
        self.assertTrue(all("==" in dependency for dependency in runtime_dependencies))
        self.assertFalse({"powershap", "catboost"} & {
            _dependency_name(dependency) for dependency in runtime_dependencies
        })

    def test_dev_extra_and_requirements_entrypoint_do_not_duplicate_runtime_pins(self) -> None:
        """A second pin source can resolve a different environment than the package."""
        project = _project_metadata()
        dev_dependencies = project["optional-dependencies"]["dev"]
        requirements_lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        non_empty_lines = [line.strip() for line in requirements_lines if line.strip()]

        self.assertEqual(dev_dependencies, DEV_DEPENDENCIES)
        self.assertFalse({
            _dependency_name(dependency) for dependency in dev_dependencies
        } & {_dependency_name(dependency) for dependency in RUNTIME_DEPENDENCIES})
        self.assertEqual(non_empty_lines, ["-e .[dev]"])
        self.assertTrue(all("==" not in line for line in non_empty_lines))
