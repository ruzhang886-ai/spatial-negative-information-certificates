"""RMDirectionalBerlin 射线追踪无线电地图的轻量读取器。"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from fnmatch import fnmatch
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image


_PATH_GAIN_PATTERN = re.compile(r"^pg_(.+)_(.+)_(.+)_(.+)\.png$")


class RMDirectionalDataset:
    """统一读取已解压目录或官方ZIP压缩包。"""

    def __init__(self, source: str | Path):
        self.source = Path(source).expanduser().resolve()
        self._archive: ZipFile | None = None

    def __enter__(self) -> "RMDirectionalDataset":
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            self._archive = ZipFile(self.source)
        self.validate()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    @property
    def is_archive(self) -> bool:
        return self._archive is not None

    def validate(self) -> None:
        if self.source.is_dir():
            missing = [
                name
                for name in ("path_gain", "tx_antennas", "antenna_patterns", "gis")
                if not (self.source / name).is_dir()
            ]
            if missing:
                raise FileNotFoundError(
                    f"RMDirectionalBerlin目录不完整：{self.source}；缺少 {', '.join(missing)}。"
                )
            if not (self.source / "max_power.json").is_file():
                raise FileNotFoundError(f"缺少路径增益缩放文件：{self.source / 'max_power.json'}")
            return
        if self._archive is None:
            raise FileNotFoundError(f"数据源不是目录或ZIP文件：{self.source}")
        names = set(self._archive.namelist())
        missing = [
            name
            for name in ("path_gain", "tx_antennas", "antenna_patterns", "gis")
            if not any(member.startswith(f"{name}/") for member in names)
        ]
        if missing:
            raise FileNotFoundError(
                f"RMDirectionalBerlin压缩包不完整：{self.source}；缺少 {', '.join(missing)}。"
            )
        if "max_power.json" not in names:
            raise FileNotFoundError(f"压缩包缺少 max_power.json：{self.source}")

    def names(self, prefix: str) -> list[str]:
        if self._archive is not None:
            return [name for name in self._archive.namelist() if fnmatch(name, prefix)]
        root = self.source / Path(prefix).parent
        return [f"{Path(prefix).parent.as_posix()}/{path.name}" for path in root.glob(Path(prefix).name)]

    def read_bytes(self, relative_path: str) -> bytes:
        if self._archive is not None:
            try:
                return self._archive.read(relative_path)
            except KeyError as exc:
                raise FileNotFoundError(f"压缩包中缺少 {relative_path}") from exc
        path = self.source / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()


@contextmanager
def open_dataset(source: str | Path | RMDirectionalDataset):
    """打开一次数据源，批量读取ZIP时避免重复扫描中央目录。"""
    if isinstance(source, RMDirectionalDataset):
        yield source
        return
    with RMDirectionalDataset(source) as dataset:
        yield dataset


@dataclass(frozen=True)
class RMDirectionalSample:
    """一个发射配置及其256×256地面路径增益图。"""

    sample_id: tuple[str, str, str, str]
    path_gain_db: np.ndarray
    tx_coords: np.ndarray
    tx_azimuth_rad: float
    tx_tilt_rad: float
    antenna_id: int
    tx_parameters: dict
    building_height_m: np.ndarray


def validate_dataset_root(dataset_root: str | Path) -> Path:
    source = Path(dataset_root).expanduser().resolve()
    with RMDirectionalDataset(source):
        pass
    return source


def discover_sample_ids(
    dataset_root: str | Path | RMDirectionalDataset,
) -> list[tuple[str, str, str, str]]:
    with open_dataset(dataset_root) as dataset:
        result: list[tuple[str, str, str, str]] = []
        for name in sorted(dataset.names("path_gain/pg_*.png")):
            match = _PATH_GAIN_PATTERN.match(Path(name).name)
            if match:
                result.append(tuple(match.groups()))
        if not result:
            raise FileNotFoundError(f"{dataset.source} 中没有找到 path_gain/pg_*.png。")
        return result


def load_power_scale(
    dataset_root: str | Path | RMDirectionalDataset,
) -> tuple[float, float]:
    with open_dataset(dataset_root) as dataset:
        payload = json.loads(dataset.read_bytes("max_power.json").decode("utf-8"))
    try:
        lower = float(payload["pg_trnc"])
        upper = float(payload["pg_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("max_power.json 必须包含数值 pg_trnc 和 pg_max。") from exc
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"无效的路径增益范围：[{lower}, {upper}] dB。")
    return lower, upper


def decode_path_gain_db(image: np.ndarray, lower_db: float, upper_db: float) -> np.ndarray:
    """把数据集的8位灰度路径增益图还原到dB尺度。"""
    values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError("路径增益图必须是二维灰度图。")
    normalized = values.astype(float) / 255.0
    return lower_db + normalized * (upper_db - lower_db)


def _select_tx_parameters(payload, index_token: str) -> dict:
    if isinstance(payload, list):
        index = int(index_token)
        try:
            selected = payload[index]
        except IndexError as exc:
            raise IndexError(f"发射配置索引 {index} 超出JSON列表范围。") from exc
    elif isinstance(payload, dict):
        if index_token in payload:
            selected = payload[index_token]
        elif str(int(index_token)) in payload:
            selected = payload[str(int(index_token))]
        else:
            raise KeyError(f"JSON中不存在发射配置索引 {index_token}。")
    else:
        raise TypeError("发射配置JSON必须是列表或字典。")
    if not isinstance(selected, dict):
        raise TypeError("选中的发射配置必须是JSON对象。")
    return selected


def load_sample(
    dataset_root: str | Path | RMDirectionalDataset,
    sample_id: tuple[str, str, str, str],
) -> RMDirectionalSample:
    if len(sample_id) != 4:
        raise ValueError("样本编号必须包含四段。")
    prefix = "_".join(sample_id[:3])
    image_path = f"path_gain/pg_{'_'.join(sample_id)}.png"
    params_path = f"tx_antennas/{prefix}_txparams.json"
    building_path = f"gis/nbuildings_{prefix}.png"
    with open_dataset(dataset_root) as dataset:
        lower_db, upper_db = load_power_scale(dataset)
        with Image.open(BytesIO(dataset.read_bytes(image_path))) as image:
            path_gain_db = decode_path_gain_db(
                np.asarray(image.convert("L")), lower_db, upper_db
            )
        payload = json.loads(dataset.read_bytes(params_path).decode("utf-8"))
        with Image.open(BytesIO(dataset.read_bytes(building_path))) as image:
            building_height_m = np.asarray(image.convert("L"), dtype=float)
    if building_height_m.shape != path_gain_db.shape:
        raise ValueError(
            f"建筑高度图与路径增益图尺寸不一致：{building_height_m.shape} != {path_gain_db.shape}"
        )
    parameters = _select_tx_parameters(payload, sample_id[3])
    try:
        tx_coords = np.asarray(parameters["tx_coords"], dtype=float).reshape(3)
        azimuth = float(parameters["phi"])
        tilt = float(parameters["theta"])
        antenna_id = int(parameters["antenna"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{Path(params_path).name} 中的配置缺少 tx_coords/phi/theta/antenna。"
        ) from exc
    if not np.all(np.isfinite(tx_coords)):
        raise ValueError("发射位置包含非有限值。")
    return RMDirectionalSample(
        sample_id=tuple(sample_id),
        path_gain_db=path_gain_db,
        tx_coords=tx_coords,
        tx_azimuth_rad=azimuth,
        tx_tilt_rad=tilt,
        antenna_id=antenna_id,
        tx_parameters=parameters,
        building_height_m=building_height_m,
    )


def nearest_path_gain_db(sample: RMDirectionalSample, point) -> float | None:
    """返回最近像素的路径增益；越出地图时返回None。"""
    point = np.asarray(point, dtype=float).reshape(2)
    row, column = np.rint(point).astype(int)
    height, width = sample.path_gain_db.shape
    if row < 0 or column < 0 or row >= height or column >= width:
        return None
    return float(sample.path_gain_db[row, column])


def nearest_building_height_m(sample: RMDirectionalSample, point) -> float | None:
    """返回最近像素的建筑高度；越出地图时返回None。"""
    point = np.asarray(point, dtype=float).reshape(2)
    row, column = np.rint(point).astype(int)
    height, width = sample.building_height_m.shape
    if row < 0 or column < 0 or row >= height or column >= width:
        return None
    return float(sample.building_height_m[row, column])
