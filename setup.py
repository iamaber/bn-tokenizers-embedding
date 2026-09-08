"""Build a platform wheel containing a Go shared library, without a CPython ABI."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()


class GoBuildExt(build_ext):
    def get_ext_filename(self, fullname: str) -> str:
        suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
        return os.path.join(*fullname.split(".")) + suffix

    def build_extension(self, ext: Extension) -> None:
        go = os.environ.get("BNTOK_GO") or shutil.which("go")
        if not go:
            raise RuntimeError(
                "Building from source requires Go 1.24+; set BNTOK_GO or install Go."
            )
        output = Path(self.get_ext_fullpath(ext.name)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                go,
                "build",
                "-trimpath",
                "-buildmode=c-shared",
                "-ldflags=-s -w",
                "-o",
                str(output),
                "./cmd/bntok-shared",
            ],
            cwd=ROOT,
            env={**os.environ, "CGO_ENABLED": "1"},
            check=True,
        )
        output.with_suffix(".h").unlink(missing_ok=True)


class PlatformWheel(bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    ext_modules=[Extension("bn_tokenizers_embedding._go", sources=[])],
    cmdclass={"build_ext": GoBuildExt, "bdist_wheel": PlatformWheel},
)
