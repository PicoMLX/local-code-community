#!/bin/sh
# Bazel workspace status for the native Local Code library.
set -eu

prefix=$(git rev-parse --show-prefix 2>/dev/null) || exit 0
[ -z "$prefix" ] || exit 0
revision=$(git rev-parse --verify HEAD 2>/dev/null) || exit 0
tag=$(git describe --tags --exact-match HEAD 2>/dev/null) || tag=
dirty=
if status=$(git --no-optional-locks status --porcelain --ignore-submodules=none 2>/dev/null); then
    if [ -n "$status" ]; then dirty=true; else dirty=false; fi
fi

# Hex escapes keep the stable status on one line and are safe in a C literal.
/usr/bin/python3 - "$revision" "$tag" "$dirty" <<'PY'
import json
import sys

revision, tag, dirty = sys.argv[1:]
data = {"revision": revision}
if tag:
    data["tag"] = tag
if dirty:
    data["dirty"] = dirty == "true"
encoded = json.dumps(data, separators=(",", ":")).encode("ascii")
print("STABLE_PICO_NATIVE_BUILD_INFO " + "".join(f"\\x{byte:02x}" for byte in encoded))
PY
