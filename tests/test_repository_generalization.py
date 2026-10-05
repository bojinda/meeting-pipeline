"""Keep public examples synthetic without storing historical/private identities."""
from __future__ import annotations

import ast
import calendar
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
APPROVED_NAMES = frozenset({"Morgan", "Mórgan", "Taylor", "Casey", "Riley", "Chair"})
SYNTHETIC_NAMES = re.compile(r"\b(?:" + "|".join(APPROVED_NAMES - {"Chair"}) + r")\b", re.IGNORECASE)
PLACEHOLDERS = frozenset({"Unknown", "Unclear", "Unidentified", "None", "Unassigned", "Name", "Owner", "I", "We", "They", "You", "It", "That", "This"})
QUALIFIERS = frozenset({"Possibly", "Maybe", "Apparently", "Probably"})
MONTH_NAMES = {name.lower() for name in (*calendar.month_name[1:], *calendar.month_abbr[1:])} | {"sept"}
HISTORICAL_LABEL = re.compile(r"(?:^|[_ -])(?:" + "|".join(sorted(MONTH_NAMES)) + r")(?=$|[_ -])|\b\d{4}[-_]\d{1,2}[-_]\d{1,2}\b|(?:^|_)meeting[_ -]?\d+", re.IGNORECASE)
GPU_UUID = re.compile(r"\bGPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
GPU_UUID_LOCK = re.compile(r"\baihub-gpu-[0-9a-f]{8}\.lock\b", re.IGNORECASE)
_PERSON = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*){0,2}(?![\w'’-])"
PERSON_CONTEXTS = (
    re.compile(r"\[(" + _PERSON + r")\]\s+(?=[A-Za-z])"),
    re.compile(r"(?i:\b(?:motion by|moved by|seconded by|mover|seconder)\s*:?)[*_]*\s+(" + _PERSON + r")"),
    re.compile(r"(?i:\b(?:contact|contacted|messaged|assigned to))\s+(" + _PERSON + r")"),
    re.compile(r"\b(" + _PERSON + r"),\s+(?i:I think|we probably)\b"),
    re.compile(r"\b(" + _PERSON + r")\s+(?i:reported|will|agreed to|moved|seconded)\b"),
    re.compile(r"\b(?:Mr|Ms|Mrs|Dr)\.\s+(" + _PERSON + r")"),
)


def public_example_files(root: Path = ROOT) -> list[Path]:
    paths = [root / "README.md", root / "TROUBLESHOOTING.md"]
    paths += list((root / "prompts").rglob("*.txt"))
    paths += list((root / "tests" / "fixtures").glob("*.json"))
    paths += list((root / "tests").glob("test_*.py"))
    for directory in ("bin", "controller"):
        for suffix in ("*.py", "*.sh"):
            paths += list((root / directory).rglob(suffix))
    paths += list((root / "config").glob("*.example"))
    paths += list((root / "config").glob("*.example.json"))
    paths += list((root / "systemd").glob("*.example"))
    return [path for path in paths if path.is_file()]


def private_identifier_violations(root: Path = ROOT, identifier_file: Path | None = None) -> list[str] | None:
    path = identifier_file if identifier_file is not None else root / "tests" / "private-identifiers.txt"
    if not path.exists():
        return None
    identifiers = {line.strip().casefold() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")}
    locations = []
    for public in public_example_files(root):
        for number, line in enumerate(public.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if any(identifier in line.casefold() for identifier in identifiers):
                locations.append(f"{public.relative_to(root).as_posix()}:{number}")
    return locations


def _approved_person(name: str) -> bool:
    name = name.strip(" .:-*_[]()")
    if re.fullmatch(r"SPEAKER_\d+", name) or name in PLACEHOLDERS or name in QUALIFIERS:
        return True
    return all(part in APPROVED_NAMES for part in name.split())


class RepositoryGeneralizationTests(unittest.TestCase):
    def test_public_participant_examples_use_approved_synthetic_names(self):
        violations = []
        for path in public_example_files():
            for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
                for pattern in PERSON_CONTEXTS:
                    if any(not _approved_person(match[1]) for match in pattern.finditer(line)):
                        violations.append(f"{path.relative_to(ROOT)}:{number}")
        for path in (ROOT / "config").glob("speaker_aliases*.example.json"):
            if any(not _approved_person(name) for name in json.loads(path.read_text(encoding="utf-8")).values()):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(sorted(set(violations)), [], "Use synthetic participants in public examples: " + ", ".join(sorted(set(violations))))

    def test_production_rules_have_no_fixed_participant_names(self):
        violations = []
        identities = {"speaker", "owner", "mover", "seconder", "participant", "addressee", "identity"}
        for directory in ("bin", "controller"):
            for path in (ROOT / directory).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in ast.walk(tree):
                    body = getattr(node, "body", None)
                    if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                        del body[0]
                for node in ast.walk(tree):
                    fixed = isinstance(node, ast.Constant) and isinstance(node.value, str) and SYNTHETIC_NAMES.search(node.value)
                    if isinstance(node, ast.Compare):
                        fields = {value.id for value in ast.walk(node) if isinstance(value, ast.Name)} | {value.attr for value in ast.walk(node) if isinstance(value, ast.Attribute)}
                        operands = [node.left, *node.comparators]
                        values = [value for operand in operands for value in (operand.elts if isinstance(operand, (ast.Tuple, ast.List, ast.Set)) else [operand])]
                        literals = [value.value for value in values if isinstance(value, ast.Constant) and isinstance(value.value, str)]
                        fixed = fixed or bool(fields & identities and any(value and value not in {"UNKNOWN", "UNIDENTIFIED", "NONE", "N/A"} for value in literals))
                    if fixed:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(violations, [], "Participant matching must use source identities/aliases, not literals: " + ", ".join(violations))

    def test_regression_identifiers_and_fixture_names_describe_behavior(self):
        violations = []
        for path in (ROOT / "tests" / "fixtures").glob("*"):
            if HISTORICAL_LABEL.search(path.stem):
                violations.append(str(path.relative_to(ROOT)))
        for path in (ROOT / "tests").glob("test_*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and HISTORICAL_LABEL.search(node.name):
                    violations.append(f"{path.relative_to(ROOT)}:{node.name}")
        self.assertEqual(violations, [], "Regression names must identify behavior, not a historical meeting")

    def test_public_files_do_not_embed_physical_gpu_identifiers(self):
        violations = []
        for path in public_example_files():
            for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
                if GPU_UUID.search(line) or GPU_UUID_LOCK.search(line):
                    violations.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(violations, [], "Configure stable installation-specific paths privately: " + ", ".join(violations))

    def test_public_gpu_configuration_defaults_are_logical_resources(self):
        settings = (ROOT / "config" / ".env.example").read_text(encoding="utf-8")
        for setting in ("AIHUB_GPU0_LOCK_FILE=/tmp/aihub-gpu0.lock", "AIHUB_GPU1_LOCK_FILE=/tmp/aihub-gpu1.lock", "AIHUB_GPU_LOCK_TIMEOUT=3600", "WHISPERX_DEVICE_INDEX=0", "WHISPERX_DEVICE=cuda"):
            self.assertIn(setting, settings.splitlines())

    def test_optional_private_identifier_audit(self):
        violations = private_identifier_violations()
        if violations is None:
            self.skipTest("No optional local private identifier file")
        self.assertEqual(violations, [], "Private identifier matches at: " + ", ".join(violations))

    def test_private_identifier_file_is_ignored_and_untracked(self):
        git = shutil.which("git")
        if not git:
            self.skipTest("Git is unavailable")
        ignored = subprocess.run([git, "check-ignore", "--no-index", "tests/private-identifiers.txt"], cwd=ROOT, capture_output=True, text=True)
        tracked = subprocess.run([git, "ls-files", "--error-unmatch", "tests/private-identifiers.txt"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(ignored.returncode, 0, ignored.stderr)
        self.assertNotEqual(tracked.returncode, 0, "The local identifier audit file must never be tracked")


class OptionalPrivateAuditTests(unittest.TestCase):
    def test_absent_file_requires_no_private_data(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(private_identifier_violations(Path(directory)))

    def test_temporary_file_ignores_blanks_comments_and_matches_case_insensitively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            private = root / "tests" / "private-identifiers.txt"
            private.write_text("\n# ignored comment\n   # another comment\nConfidential Demo Token\n\n", encoding="utf-8")
            (root / "README.md").write_text("# Synthetic example\nCONFIDENTIAL demo TOKEN\nPublic text\nconfidential DEMO token\n", encoding="utf-8")
            locations = private_identifier_violations(root)
            self.assertEqual(locations, ["README.md:2", "README.md:4"])
            self.assertTrue(all("Confidential Demo Token" not in location for location in locations))

    def test_identifiers_are_literal_and_empty_configuration_is_a_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "local-audit.txt"
            private.write_text("demo+marker\n", encoding="utf-8")
            (root / "README.md").write_text("demomarker\ndemo+MARKER\n", encoding="utf-8")
            self.assertEqual(private_identifier_violations(root, private), ["README.md:2"])
            private.write_text("# comments only\n\n", encoding="utf-8")
            self.assertEqual(private_identifier_violations(root, private), [])

    def test_private_audit_covers_public_surfaces_without_reading_private_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            surfaces = ["README.md", "TROUBLESHOOTING.md", "prompts/meeting/example.txt", "bin/example.py", "controller/example.py", "config/.env.example", "config/speaker_aliases.example.json", "systemd/example.service.example", "tests/test_example.py", "tests/fixtures/behavior.json"]
            for name in surfaces:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Audit Demo Marker\n", encoding="utf-8")
            (root / "config" / ".env").write_text("Audit Demo Marker\n", encoding="utf-8")
            private = root / "tests" / "private-identifiers.txt"
            private.write_text("audit demo marker\n", encoding="utf-8")
            self.assertEqual(set(private_identifier_violations(root)), {name + ":1" for name in surfaces})


if __name__ == "__main__":
    unittest.main()
