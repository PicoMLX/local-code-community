"""Generate the native source export from Bazel's stable workspace status."""

def _pico_build_info_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".c")
    ctx.actions.run_shell(
        inputs = [ctx.info_file, ctx.file.generator],
        outputs = [output],
        arguments = [ctx.file.generator.path, ctx.info_file.path, output.path],
        command = '/usr/bin/python3 "$1" --status-file "$2" --output "$3"',
        mnemonic = "PicoBuildInfo",
    )
    return [DefaultInfo(files = depset([output]))]

pico_build_info = rule(
    implementation = _pico_build_info_impl,
    attrs = {
        "generator": attr.label(
            default = "//Scripts:pico_build_info.py",
            allow_single_file = True,
        ),
    },
)
