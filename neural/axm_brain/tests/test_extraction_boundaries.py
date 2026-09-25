import ast
import hashlib
import json
from pathlib import Path
import unittest

from neural.axm_brain import AXMBrain, BrainConfig

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "provenance" / "direct-brain-donor-v0.1.json"

def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()

class ExtractionBoundaryTests(unittest.TestCase):
    def test_donor_provenance_distinguishes_exact_copies_from_adaptations(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["donor_commit"],
            "163410ce05a42879dca478d8ea0de7f6d17c4820",
        )
        for relpath, expected_sha in manifest["exact_copy_files"].items():
            self.assertEqual(git_blob_sha(ROOT / relpath), expected_sha, relpath)

        adaptations = manifest["adapted_from_donor"]
        self.assertTrue(adaptations)
        for relpath, record in adaptations.items():
            self.assertNotEqual(
                record["donor_blob_sha"],
                record["current_blob_sha"],
                relpath,
            )
            self.assertEqual(
                git_blob_sha(ROOT / relpath),
                record["current_blob_sha"],
                relpath,
            )
            self.assertTrue(record["reason"].strip(), relpath)

    def test_runtime_core_has_no_ambient_authority_imports(self):
        forbidden = {
            "os", "pathlib", "shutil", "socket", "subprocess",
            "requests", "urllib", "http", "ftplib", "telnetlib",
        }
        runtime_files = [
            "core.py", "state.py", "contract.py",
            "taxonomy.py", "roots.py", "analyzer.py",
        ]
        for filename in runtime_files:
            path = ROOT / "neural" / "axm_brain" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            self.assertFalse(forbidden & imports, f"{filename}: {forbidden & imports}")

    def test_deterministic_birth_survives_dedicated_extraction(self):
        cfg = BrainConfig(input_size=2, hidden_size=4, output_size=1, seed=73)
        self.assertEqual(AXMBrain(cfg).to_snapshot(), AXMBrain(cfg).to_snapshot())

if __name__ == "__main__":
    unittest.main()
