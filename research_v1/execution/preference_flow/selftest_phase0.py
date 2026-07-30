"""Pure unit tests for preference-flow Phase 0.

The tests use repository-contained read-only fixtures and imports of the
existing StylePlanner implementation.  They do not require a mounted server
dataset or execute a planner rollout.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from research_v1.execution.preference_flow.phase0_contracts import (
    A3_8_CHECKPOINT_ROOT_DEFAULT,
    SERVER_RECORD_ROOT_DEFAULT,
    SERVER_REPO_ROOT_DEFAULT,
    Phase0ContractError,
    parse_rho_grid,
    resolve_phase0_paths,
    run_phase0_contract_checks,
)
from research_v1.execution.preference_flow.phase0_manifest import (
    ManifestRequest,
    build_manifest,
)


class Phase0ContractsTest(unittest.TestCase):
    def test_defaults_are_server_paths(self) -> None:
        self.assertEqual(
            SERVER_REPO_ROOT_DEFAULT.as_posix(),
            "/home/lisw/programs/Nuplan-Diffusion-Baseline",
        )
        self.assertEqual(
            SERVER_RECORD_ROOT_DEFAULT.as_posix(),
            "/mnt/mydata/lishangwen/Nuplan-Baseline-Record",
        )
        self.assertEqual(
            A3_8_CHECKPOINT_ROOT_DEFAULT.as_posix(),
            "/mnt/mydata/lishangwen/Nuplan-Baseline-Record/preference_flow_v1/"
            "checkpoints/a3_8",
        )

    def test_rho_grid_requires_neutral_and_in_range(self) -> None:
        self.assertEqual(parse_rho_grid("-1,0,1"), (-1.0, 0.0, 1.0))
        with self.assertRaises(Phase0ContractError):
            parse_rho_grid("-1,1")
        with self.assertRaises(Phase0ContractError):
            parse_rho_grid("0,1.01")

    def test_existing_baseline_contracts(self) -> None:
        report = run_phase0_contract_checks()
        self.assertTrue(report["passed"], msg=json.dumps(report, ensure_ascii=False, indent=2))


class Phase0ManifestTest(unittest.TestCase):
    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _request(self, *, dry_run: bool, use_missing_base: bool = False) -> ManifestRequest:
        # Keep the self-test fully read-only: existing package files exercise
        # hashing and dry-run behavior without creating a temporary directory.
        artifacts = Path(__file__).resolve().parent
        base_path = artifacts / "missing-base.pth" if use_missing_base else artifacts / "phase0_contracts.py"
        paths = resolve_phase0_paths(
            repo_root=self._repo_root(),
            record_root=artifacts / "read-only-record-root",
            a3_8_checkpoint_root=artifacts / "read-only-a3-8-root",
            dataset_split_path=artifacts / "README.md",
            smoke_cohort_path=artifacts / "README.md",
            manifest_path=artifacts / "not_written_phase0_manifest.json",
        )
        return ManifestRequest(
            paths=paths,
            base_checkpoint_path=str(base_path),
            style_checkpoint_path=str(artifacts / "phase0_manifest.py"),
            calibration_paths=(str(artifacts / "phase0_contracts.py"),),
            reference_paths=(str(artifacts / "selftest_phase0.py"),),
            config_paths=(str(artifacts / "README.md"),),
            smoke_cohort_path=str(artifacts / "README.md"),
            seed=3407,
            rho_grid=(-1.0, 0.0, 1.0),
            dry_run=dry_run,
            strict=False,
            actual_command="phase0-selftest",
        )

    def test_non_dry_manifest_hashes_real_artifacts(self) -> None:
        request = self._request(dry_run=False)
        manifest = build_manifest(request)
        expected_hash = hashlib.sha256(
            (Path(__file__).resolve().parent / "phase0_contracts.py").read_bytes()
        ).hexdigest()
        self.assertEqual(manifest["artifacts"]["base_checkpoint"]["sha256"], expected_hash)
        self.assertEqual(manifest["artifacts"]["base_checkpoint"]["status"], "present")
        self.assertNotIn("git", manifest)
        self.assertEqual(manifest["strict_failures"], [])

    def test_dry_run_marks_missing_without_a_fake_hash(self) -> None:
        request = self._request(dry_run=True, use_missing_base=True)
        manifest = build_manifest(request)
        artifact = manifest["artifacts"]["base_checkpoint"]
        self.assertEqual(artifact["status"], "missing")
        self.assertIsNone(artifact["sha256"])
        self.assertEqual(artifact["hash_status"], "missing")
        self.assertIn("base_checkpoint: missing", manifest["strict_failures"])
        self.assertFalse(request.paths.default_manifest_path.exists())


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
