"""Generates the native build info export from Bazel's stable workspace status."""

def _native_build_info_impl(ctx):
    source = ctx.actions.declare_file(ctx.label.name + ".c")
    ctx.actions.run_shell(
        inputs = [ctx.info_file],
        outputs = [source],
        arguments = [ctx.info_file.path, source.path],
        command = """
info=$(/usr/bin/awk '$1 == "STABLE_PICO_NATIVE_BUILD_INFO" { print $2 }' "$1")
if [ -n "$info" ]; then
    value=$(printf '"%s"' "$info")
else
    value=0
fi
printf '__attribute__((visibility("default"))) const char *pico_local_code_build_info_v1(void) { return %s; }\\n' "$value" > "$2"
""",
        mnemonic = "NativeBuildInfo",
    )
    return [DefaultInfo(files = depset([source]))]

native_build_info = rule(implementation = _native_build_info_impl)
