"""Remove machine-specific data from generated MuJoCo stubs.

Adapted from mjlab's Apache-2.0-licensed stub post-processing workflow:
https://github.com/mujocolab/mjlab/tree/main/typings
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

STUB_DIR = Path(__file__).parent / "mujoco"
BACKENDS = ("cgl", "egl", "glfw", "glx", "osmesa")
REMOVE = (*BACKENDS, "gl_context.pyi", "renderer.pyi", "rendering")
PLATFORM_NAMES = ("is_rosetta", "proc_translated")

BACKEND_IMPORT = re.compile(
  rf"^(?:from mujoco\.(?:{'|'.join(BACKENDS)}) import GLContext"
  rf"|from \. import (?:{'|'.join(BACKENDS)}))$"
)
PLATFORM_DEFINITION = re.compile(rf"^(?:{'|'.join(PLATFORM_NAMES)})\b")
SYSTEM_DEFAULT = re.compile(r"^(_SYSTEM: str) = '\w+'$")
VALUE_COMMENT = re.compile(r" {2,}# value = .*$")
PATH_DEFAULT = re.compile(
  r"^(\s*\w+: str) = '[^']*(?:site-packages|/Users/|/home/)[^']*'\s*$"
)


def remove_platform_bindings() -> None:
  for name in REMOVE:
    target = STUB_DIR / name
    if target.is_dir():
      shutil.rmtree(target)
    elif target.exists():
      target.unlink()


def normalize_init() -> None:
  init_path = STUB_DIR / "__init__.pyi"
  normalized = []
  for line in init_path.read_text().splitlines():
    if BACKEND_IMPORT.match(line.strip()) or PLATFORM_DEFINITION.match(line):
      continue
    line = SYSTEM_DEFAULT.sub(r"\1", line)
    if line.startswith("__all__"):
      for name in ("GLContext", *BACKENDS, *PLATFORM_NAMES):
        line = line.replace(f"'{name}', ", "")
    normalized.append(line)
  init_path.write_text("\n".join(normalized) + "\n")


def remove_unstable_values() -> None:
  for stub_path in STUB_DIR.rglob("*.pyi"):
    normalized = []
    for line in stub_path.read_text().splitlines():
      if "# value = " in line and ("site-packages" in line or "0x" in line):
        line = VALUE_COMMENT.sub("", line)
      if match := PATH_DEFAULT.match(line):
        line = match.group(1)
      normalized.append(line)
    stub_path.write_text("\n".join(normalized) + "\n")


def main() -> None:
  remove_platform_bindings()
  normalize_init()
  remove_unstable_values()


if __name__ == "__main__":
  main()
