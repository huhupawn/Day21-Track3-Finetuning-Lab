#!/usr/bin/env python3
"""Package Lab 21 submission into Option A ZIP format:

lab21_<MSSV>/
├── submission/
│   ├── REPORT.md
│   └── REFLECTION.md
├── results/              (all json + runs.csv)
├── adapters/correct/     (adapter_config.json + adapter_model.safetensors)
└── notebooks/            (.py files)
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

def package(mssv: str) -> pathlib.Path:
    clean_mssv = mssv.strip().replace(" ", "_")
    folder_name = f"lab21_{clean_mssv}"
    out_dir = ROOT / folder_name

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1. Copy submission/
    (out_dir / "submission").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "submission" / "REPORT.md", out_dir / "submission" / "REPORT.md")
    if (ROOT / "submission" / "REFLECTION.md").exists():
        shutil.copy2(ROOT / "submission" / "REFLECTION.md", out_dir / "submission" / "REFLECTION.md")

    # 2. Copy results/
    shutil.copytree(ROOT / "results", out_dir / "results", dirs_exist_ok=True)

    # 3. Copy adapters/correct/
    shutil.copytree(ROOT / "adapters" / "correct", out_dir / "adapters" / "correct", dirs_exist_ok=True)

    # 4. Copy notebooks/
    shutil.copytree(ROOT / "notebooks", out_dir / "notebooks", dirs_exist_ok=True)

    # 5. Create ZIP archive
    zip_path = ROOT / f"{folder_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                full_path = pathlib.Path(root) / file
                rel_path = full_path.relative_to(ROOT)
                zf.write(full_path, rel_path)

    print(f"Created submission folder: {out_dir}")
    print(f"Created ZIP file: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return zip_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Lab 21 Option A submission")
    parser.add_argument("--mssv", default="2A202602816", help="Mã số sinh viên (Student ID)")
    args = parser.parse_args()
    package(args.mssv)
