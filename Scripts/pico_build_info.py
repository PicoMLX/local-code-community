#!/usr/bin/python3
"""Stamp the native library from Bazel's declared workspace status input."""

import argparse
import json
from pathlib import Path
import re
import subprocess


def source_info(directory):
    def git(*arguments):
        try:
            result = subprocess.run(["git", "-C", str(directory), *arguments],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    checkout = git("rev-parse", "--show-toplevel")
    if checkout is None or Path(checkout).resolve() != directory.resolve():
        return None
    revision = git("rev-parse", "--verify", "HEAD")
    if revision is None or not re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", revision):
        return None
    status = git("status", "--porcelain", "--untracked-files=normal")
    info = {"revision": revision}
    if status is not None:
        info["dirty"] = bool(status)
    tag = git("describe", "--tags", "--exact-match", "HEAD")
    if tag:
        info["tag"] = tag
    return info


def generate(status_file, output):
    info = None
    for line in status_file.read_text().splitlines():
        key, _, value = line.partition(" ")
        if key == "STABLE_PICO_BUILD_INFO":
            candidate = json.loads(value)
            if re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", candidate.get("revision", "")):
                info = candidate
    value = "NULL" if info is None else json.dumps(json.dumps(info, separators=(",", ":")))
    output.write_text("#include <stddef.h>\n"
                      "const char *pico_local_code_build_info_v1(void) {\n"
                      "  return " + value + ";\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.status_file is not None and args.output is not None:
        generate(args.status_file, args.output)
    elif args.status_file is None and args.output is None:
        info = source_info(Path(__file__).resolve().parents[1])
        if info is not None:
            print("STABLE_PICO_BUILD_INFO " + json.dumps(info, separators=(",", ":")))
    else:
        parser.error("--status-file and --output are required together")


if __name__ == "__main__":
    main()
