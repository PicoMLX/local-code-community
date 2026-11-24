import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "pico_build_info.py"
SPEC = importlib.util.spec_from_file_location("pico_build_info", SCRIPT)
BUILD_INFO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_INFO)


class BuildInfoTests(unittest.TestCase):
    def test_git_identity_uses_exact_tag_and_includes_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*arguments):
                return subprocess.check_output(["git", "-C", str(root), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", *arguments],
                    stderr=subprocess.DEVNULL, text=True).strip()
            git("init")
            source = root / "source.txt"
            source.write_text("original\n")
            git("add", "source.txt")
            git("commit", "-m", "fixture")
            revision = git("rev-parse", "HEAD")
            git("tag", "v1.2.3")
            self.assertEqual(BUILD_INFO.source_info(root), {"revision": revision, "tag": "v1.2.3", "dirty": False})
            archive = root / "unpacked-source-archive"
            archive.mkdir()
            self.assertIsNone(BUILD_INFO.source_info(archive))
            source.write_text("changed\n")
            self.assertTrue(BUILD_INFO.source_info(root)["dirty"])
            git("add", "source.txt")
            git("commit", "-m", "changed")
            self.assertNotIn("tag", BUILD_INFO.source_info(root))
            (root / "new-source.txt").write_text("untracked source\n")
            self.assertTrue(BUILD_INFO.source_info(root)["dirty"])

    def test_source_archive_never_invents_a_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(BUILD_INFO.source_info(Path(directory)))

    def test_unavailable_git_leaves_the_optional_identity_unknown(self):
        with patch.object(BUILD_INFO.subprocess, "run", side_effect=FileNotFoundError()):
            self.assertIsNone(BUILD_INFO.source_info(Path("/source")))

    def test_generated_c_export_reports_the_stamped_identity_and_unknown_stays_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"revision": "a" * 40, "tag": 'v1.2.3-"quoted"', "dirty": True}
            for index, info in enumerate([expected, None]):
                status = root / (str(index) + ".status")
                status.write_text("" if info is None else "STABLE_PICO_BUILD_INFO " + json.dumps(info) + "\n")
                source = root / (str(index) + ".c")
                BUILD_INFO.generate(status, source)
                library = root / (str(index) + ".dylib")
                flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
                subprocess.run(["cc", *flags, str(source), "-o", str(library)], check=True)
                export = ctypes.CDLL(str(library)).pico_local_code_build_info_v1
                export.restype = ctypes.c_char_p
                returned = export()
                self.assertEqual(None if returned is None else json.loads(returned), info)


if __name__ == "__main__":
    unittest.main()
