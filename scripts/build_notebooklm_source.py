"""Build NOTEBOOKLM_FULL_SOURCE.txt containing only pure production source files (src/*.py + main.py)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import configure_utf8_console

SOURCE_ORDER = [
    "main.py",
    "src/__init__.py",
    "src/config.py",
    "src/io_utils.py",
    "src/preprocessor.py",
    "src/contour_detector.py",
    "src/geometry.py",
    "src/postprocessor.py",
    "src/pipeline.py",
    "src/gui.py",
    "src/eda.py",
    "src/evaluator.py",
    "src/cli.py",
]


def build_notebooklm_source(output_path: Path | None = None) -> Path:
    """Concatenate all pure production Python files with clean headers into a single .txt file."""
    target = output_path or (PROJECT_ROOT / "NOTEBOOKLM_FULL_SOURCE.txt")
    sections: list[str] = [
        "================================================================================",
        "PROJECT: codyssey-a3-01 (OpenCV & NumPy Document Scanner)",
        "CONTENTS: Pure Production Source Code Only (Excludes tests/ and prompt guides)",
        "================================================================================",
        "",
    ]

    included_files = 0
    total_lines = 0

    for rel_path in SOURCE_ORDER:
        file_path = PROJECT_ROOT / rel_path
        if not file_path.exists():
            continue
        code = file_path.read_text(encoding="utf-8")
        line_count = len(code.splitlines())
        included_files += 1
        total_lines += line_count

        sections.append("################################################################################")
        sections.append(f"# FILE [{included_files:02d}]: {rel_path} ({line_count} lines)")
        sections.append("################################################################################")
        sections.append(code.rstrip())
        sections.append("")

    target.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[OK] Built {target.name}: {included_files} production files, {total_lines} total lines.")
    return target


if __name__ == "__main__":
    configure_utf8_console()
    build_notebooklm_source()
