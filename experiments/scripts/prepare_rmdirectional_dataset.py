from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile
import sys


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.rmdirectional_adapter import validate_dataset_root


EXPECTED_MD5 = "5ef27c261e00d660e3590086ac1d4675"
REQUIRED_DIRECTORIES = {"path_gain", "tx_antennas", "antenna_patterns", "gis"}
REQUIRED_FILES = {"max_power.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验RMBerlinDirectional.zip并只解压外部审查需要的文件。"
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE.parent / "external_data" / "RMBerlinDirectional",
    )
    parser.add_argument("--skip-md5", action="store_true")
    return parser.parse_args()


def file_md5(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def selected_relative_path(member_name: str) -> Path | None:
    parts = Path(member_name.replace("\\", "/")).parts
    for index, part in enumerate(parts):
        if part in REQUIRED_DIRECTORIES:
            return Path(*parts[index:])
        if part in REQUIRED_FILES:
            return Path(part)
    return None


def extract_selected(archive: Path, output: Path) -> dict[str, int]:
    output = output.resolve()
    counts = {name: 0 for name in sorted(REQUIRED_DIRECTORIES | REQUIRED_FILES)}
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = selected_relative_path(member.filename)
            if relative is None or member.is_dir():
                continue
            destination = (output / relative).resolve()
            try:
                destination.relative_to(output)
            except ValueError as exc:
                raise ValueError(f"压缩包成员越出目标目录：{member.filename}") from exc
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            counts[relative.parts[0]] += 1
    return counts


def main() -> None:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    actual_md5 = file_md5(archive)
    if not args.skip_md5 and actual_md5.lower() != EXPECTED_MD5:
        raise ValueError(
            f"压缩包MD5不匹配：实际 {actual_md5}，预期 {EXPECTED_MD5}。"
        )
    counts = extract_selected(archive, args.output)
    root = validate_dataset_root(args.output)
    metadata = {
        "dataset": "RMDirectionalBerlin",
        "version": "v3",
        "doi": "10.5281/zenodo.15721131",
        "source_archive": archive.name,
        "source_archive_md5": actual_md5,
        "zenodo_expected_md5": EXPECTED_MD5,
        "selected_file_counts": counts,
    }
    (root / "source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(root)


if __name__ == "__main__":
    main()
