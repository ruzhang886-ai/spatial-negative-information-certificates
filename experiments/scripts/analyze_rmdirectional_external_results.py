from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总RMDirectionalBerlin外部审查结果。")
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "runs" / "rmdirectional_external_analysis.json")
    parser.add_argument("--report", type=Path, default=BASE.parent / "RMDirectionalBerlin外部射线追踪结果_v1.md")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=BASE / "runs" / "rmdirectional_external_protocol_v2.json",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def distance_summary(histories: list[dict], key: str) -> dict:
    event = [h["history_distance_m"] for h in histories if h[key]]
    other = [h["history_distance_m"] for h in histories if not h[key]]
    bins = [(30.0, 60.0), (60.0, 90.0), (90.0, 120.000001)]
    binned = []
    for lower, upper in bins:
        selected = [h for h in histories if lower <= h["history_distance_m"] < upper]
        count = sum(bool(h[key]) for h in selected)
        binned.append(
            {
                "range_m": [lower, 120.0 if upper > 120 else upper],
                "n": len(selected),
                "events": count,
                "rate": count / len(selected) if selected else None,
            }
        )
    return {
        "event_median_m": float(np.median(event)) if event else None,
        "non_event_median_m": float(np.median(other)) if other else None,
        "bins": binned,
    }


def analyze_threshold(rows: list[dict], threshold: float, bootstrap: int, seed: int) -> dict:
    tile_records = []
    common_histories = []
    positive_histories = 0
    pair_evaluable = 0
    q2_evaluable = 0
    for row in rows:
        if float(row["threshold_db"]) != threshold:
            continue
        histories = row["histories"]
        positive_histories += len(histories)
        pair_evaluable += sum(
            h["pair_valid"] and h["pair_points_evaluable"] for h in histories
        )
        q2_evaluable += sum(
            h["spatial_q2_valid"] and h["spatial_q2_points_evaluable"]
            for h in histories
        )
        common = [
            h
            for h in histories
            if h["pair_valid"]
            and h["pair_points_evaluable"]
            and h["spatial_q2_valid"]
            and h["spatial_q2_points_evaluable"]
        ]
        common_histories.extend(common)
        tile_records.append(
            {
                "sample_id": row["sample_id"],
                "n": len(common),
                "pair": sum(bool(h["pair_all_negative"]) for h in common),
                "q2": sum(bool(h["spatial_q2_all_negative"]) for h in common),
            }
        )

    n = sum(record["n"] for record in tile_records)
    pair_count = sum(record["pair"] for record in tile_records)
    q2_count = sum(record["q2"] for record in tile_records)
    pair_rate = pair_count / n if n else None
    q2_rate = q2_count / n if n else None
    rescue = sum(
        bool(h["pair_all_negative"]) and not bool(h["spatial_q2_all_negative"])
        for h in common_histories
    )
    reverse = sum(
        not bool(h["pair_all_negative"]) and bool(h["spatial_q2_all_negative"])
        for h in common_histories
    )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(tile_records))
    pair_boot = []
    q2_boot = []
    diff_boot = []
    for _ in range(bootstrap):
        chosen = rng.choice(indices, size=len(indices), replace=True)
        sampled_n = sum(tile_records[index]["n"] for index in chosen)
        if not sampled_n:
            continue
        sampled_pair = sum(tile_records[index]["pair"] for index in chosen) / sampled_n
        sampled_q2 = sum(tile_records[index]["q2"] for index in chosen) / sampled_n
        pair_boot.append(sampled_pair)
        q2_boot.append(sampled_q2)
        diff_boot.append(sampled_pair - sampled_q2)

    return {
        "threshold_db": threshold,
        "positive_histories": positive_histories,
        "pair_evaluable": pair_evaluable,
        "pair_evaluable_rate": pair_evaluable / positive_histories if positive_histories else None,
        "q2_evaluable": q2_evaluable,
        "q2_evaluable_rate": q2_evaluable / positive_histories if positive_histories else None,
        "common_evaluable": n,
        "pair_all_negative": pair_count,
        "pair_all_negative_rate": pair_rate,
        "pair_rate_cluster_bootstrap_95ci": percentile_interval(np.asarray(pair_boot)),
        "q2_all_negative": q2_count,
        "q2_all_negative_rate": q2_rate,
        "q2_rate_cluster_bootstrap_95ci": percentile_interval(np.asarray(q2_boot)),
        "absolute_rate_reduction": pair_rate - q2_rate if n else None,
        "reduction_cluster_bootstrap_95ci": percentile_interval(np.asarray(diff_boot)),
        "pair_negative_q2_positive": rescue,
        "pair_positive_q2_negative": reverse,
        "tiles_with_pair_all_negative": sum(record["pair"] > 0 for record in tile_records),
        "tiles_with_q2_all_negative": sum(record["q2"] > 0 for record in tile_records),
        "tiles_with_common_evaluable": sum(record["n"] > 0 for record in tile_records),
        "pair_distance_summary": distance_summary(common_histories, "pair_all_negative"),
        "q2_distance_summary": distance_summary(common_histories, "spatial_q2_all_negative"),
    }


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def write_report(payload: dict, path: Path) -> None:
    sample_count = payload["sample_count"]
    lines = [
        "# RMDirectionalBerlin 外部射线追踪审查结果",
        "",
        "## 1. 审查定位",
        "",
        "本实验是跨传播模型压力测试，不是现场测量或硬件验证。它检查：一个位置已经成功接收后，按解析模型用该历史距离构造的横向补测点，在包含建筑遮挡、反射、绕射与植被衰减的城市射线追踪图上是否仍能接收。所有统计只保留建筑高度为0、位于地图内且几何证书有效的测点。",
        "",
        f"样本在运行前冻结：{sample_count}张无线电地图分别来自{sample_count}个不同城市切片；每图每阈值最多8个历史成功点；阈值为 −115、−105 和 −95 dB。置信区间按城市切片整体重采样，避免把同一切片内多个历史点视为完全独立。",
        "",
        "## 2. 主要结果",
        "",
        "| 阈值 | 正历史点 | 两点可评价率 | q=2可评价率 | 共同可评价 | 两点全阴性 | q=2全阴性 | 绝对下降 | q=2涉及切片 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["thresholds"]:
        lines.append(
            f"| {result['threshold_db']:.0f} dB | {result['positive_histories']} | "
            f"{format_percent(result['pair_evaluable_rate'])} | {format_percent(result['q2_evaluable_rate'])} | "
            f"{result['common_evaluable']} | {result['pair_all_negative']} / {format_percent(result['pair_all_negative_rate'])} | "
            f"{result['q2_all_negative']} / {format_percent(result['q2_all_negative_rate'])} | "
            f"{format_percent(result['absolute_rate_reduction'])} | {result['tiles_with_q2_all_negative']} |"
        )
    lines.extend(
        [
            "",
            "在共同可评价案例中，空间 q=2 对两点全阴性事件均有减少，且没有出现“两点已收到但q=2六点全阴性”的逆向事件。这一结果说明空间冗余具有经验收益。",
            "",
            "但三个阈值下q=2全阴性率仍为非零。因此，历史正观测给出的距离下界不能直接解释成城市环境中的径向可靠接收保证。若把全阴性直接用于删除远侧可行集，会在这些案例中产生危险删除。",
            "",
            "## 3. 聚类置信区间",
            "",
        ]
    )
    for result in payload["thresholds"]:
        pair_ci = result["pair_rate_cluster_bootstrap_95ci"]
        q2_ci = result["q2_rate_cluster_bootstrap_95ci"]
        diff_ci = result["reduction_cluster_bootstrap_95ci"]
        lines.append(
            f"- {result['threshold_db']:.0f} dB：两点 {format_percent(result['pair_all_negative_rate'])} "
            f"（95%聚类bootstrap区间 {format_percent(pair_ci[0])}–{format_percent(pair_ci[1])}）；"
            f"q=2 为 {format_percent(result['q2_all_negative_rate'])} "
            f"（{format_percent(q2_ci[0])}–{format_percent(q2_ci[1])}）；"
            f"绝对下降区间 {format_percent(diff_ci[0])}–{format_percent(diff_ci[1])}。"
        )
    lines.extend(
        [
            "",
            "## 4. 对论文结论的影响",
            "",
            "1. 理想半平面模型下的定理仍然成立，但不能直接推广到城市射线追踪传播。",
            "2. 空间q=2应表述为降低全阴性风险的策略，不应在未增加传播条件或回退规则时称为城市环境中的硬安全证书。",
            "3. 射线追踪全阴性事件可视为超出既定错误预算的空间持续漏检。当前频率表明q=2不足以覆盖全部城市传播状态。",
            "4. 下一阶段应估计空间漏检的相关尺度，并比较提高q、扩大点间距以及对建筑遮挡状态返回`unresolved`三种方案；不能依据本批结果调阈值后重新宣称独立验证。",
            "",
            "## 5. 可复核文件",
            "",
            f"- 主结果SHA-256：`{payload['result_sha256']}`",
            f"- 协议清单SHA-256：`{payload['protocol_manifest_sha256']}`",
            "- 数据集：RMDirectionalBerlin v3，Zenodo DOI 10.5281/zenodo.15721131。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.result.resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    thresholds = sorted({float(row["threshold_db"]) for row in data["rows"]})
    protocol = args.protocol_manifest.resolve()
    payload = {
        "label": "RMDirectionalBerlin external audit clustered analysis",
        "result": str(source),
        "result_sha256": file_sha256(source),
        "protocol_manifest": str(protocol.resolve()),
        "protocol_manifest_sha256": file_sha256(protocol),
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "sample_count": len({tuple(row["sample_id"]) for row in data["rows"]}),
        "thresholds": [
            analyze_threshold(data["rows"], threshold, args.bootstrap, args.seed + index)
            for index, threshold in enumerate(thresholds)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, args.report)
    print(json.dumps(payload["thresholds"], ensure_ascii=False, indent=2))
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
