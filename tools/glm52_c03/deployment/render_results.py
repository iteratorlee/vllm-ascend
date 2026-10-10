"""Render measured C03 comparisons without confidence or synthetic ITL claims."""

import argparse
import json
from pathlib import Path


def values(run, group, metric):
    if metric == "server TPOT":
        source = run["server_tpot_summary"][group]
        return [source[f"p{p}_ms"] for p in (50, 75, 90, 99)]
    if metric == "client TPOT":
        source = run["summary"][group]
        return [source[f"tpot_p{p}_ms"] for p in (50, 75, 90, 99)]
    source = run["summary"]["itl"][group]["server_itl" if metric == "server ITL" else "client_itl"]
    return [source[f"p{p}_ms"] for p in (50, 75, 90, 99)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    runs = data["controls"] + data["optimized"]
    assert len(runs) == 4
    labels = ["A2", "A3", "B0", "B1"]
    lines = [
        "# C03：完整 PD A/B 实测结果",
        "",
        "日期：2026-10-10。A 为 C02 on、原生 LI 双写；B 为 C02 on、融合 LI 双写。",
        "负载保持 131072 输入 / 1024 输出、64 请求、客户端并发 16，独立 D cache salt。",
        "P PP2/TP16；D DP2/TP16/EP32；MTP5 eager，target FULL_DECODE_ONLY。",
        "正式阶段不执行其他算子测试、microbenchmark 或 profiler。开发轮 A0/A1 不参与主比较。",
        "",
        "## 运行与饱和样本",
        "",
        "| 轮次 | 成功/失败 | D 峰值 | 峰值覆盖≥90%的请求数 | KV 峰值 %（D0/D1） | 墙钟 s | 输出 token/s |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for label, run in zip(labels, runs):
        summary = run["summary"]
        assert summary["completed"] == 64 and summary["failed"] == 0
        peak = "+".join(str(int(v)) for v in summary["observed_peak_per_node"])
        kv = "/".join(f"{v * 100:.3f}" for v in summary["kv_usage_max_per_node"])
        lines.append(
            f"| {label} | {summary['completed']}/{summary['failed']} | {peak} | "
            f"{summary['qualified_requests']} | {kv} | {run['wall_s']:.3f} | {run['output_tokens_per_wall_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "饱和子集按客户端 decode span 中采样 Running 总数达到本轮实测峰值的时间占比≥90%筛选。",
            "表内 ITL 包含这些请求的原始事件；不是只截取某个瞬间的 batch，也不把 graph token bucket 当作驻留并发。",
            "输出吞吐为本轮 65536 个输出 token / 正式请求墙钟，包含 PD 调度时间。",
        ]
    )
    for group, title in (("all_requests", "全部64请求"), ("peak_load_requests", "饱和请求子集")):
        lines.extend(
            [
                "",
                f"## {title}：TPOT / ITL 的 p50/p75/p90/p99",
                "",
                "TPOT 单位 ms/token；ITL 单位 ms/真实输出事件。所有数值保留到3位小数。",
                "",
                "| 指标 | 分位数 | A2 | A3 | B0 | B1 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in ("server TPOT", "client TPOT", "server ITL", "client ITL"):
            for index, percentile in enumerate((50, 75, 90, 99)):
                cells = " | ".join(f"{values(run, group, metric)[index]:.3f}" for run in runs)
                lines.append(f"| {metric} | p{percentile} | {cells} |")
        lines.extend(["", "| 原始样本 | A2 | A3 | B0 | B1 |", "| --- | ---: | ---: | ---: | ---: |"])
        for side, label in (("server_itl", "server ITL事件数"), ("client_itl", "client ITL事件数")):
            cells = " | ".join(str(run["summary"]["itl"][group][side]["observations"]) for run in runs)
            lines.append(f"| {label} | {cells} |")
    lines.extend(
        [
            "",
            "## MTP接受率：排除warmup的正式counter增量",
            "",
            "| 轮次 | 实例 | accepted draft token | drafted token | accepted/drafted % |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for label, run in zip(labels, runs):
        for node in run["mtp_formal_delta_per_node"]:
            lines.append(
                f"| {label} | D{node['decoder']} | {node['accepted_tokens']:.0f} | {node['drafted_tokens']:.0f} | "
                f"{node['accepted_fraction'] * 100:.4f} |"
            )
        accepted = sum(node["accepted_tokens"] for node in run["mtp_formal_delta_per_node"])
        drafted = sum(node["drafted_tokens"] for node in run["mtp_formal_delta_per_node"])
        lines.append(f"| {label} | 合计 | {accepted:.0f} | {drafted:.0f} | {accepted / drafted * 100:.4f} |")
    lines.extend(
        [
            "",
            "## 口径与局限",
            "",
            "- server TPOT = (末次−首次正token engine timestamp)/(1024−1)，由真实记录派生，非内置TPOT metric。",
            "- client TPOT沿用现有压测：(SSE完成−首次可见输出)/(1024−1)。",
            "- server ITL直接使用vLLM原metric对应事件间隔；client ITL为非空输出SSE到达间隔。MTP burst不按token均摊。",
            "- 两台D的ITL histogram增量count/sum均与正式请求原始事件校验一致；每请求均校验1024个生成token。",
            "- 每阶段两轮，保留分轮结果；事件相关，不声称统计显著性。所有四组A/B变化百分比见完整JSON。",
            "- 局部writer有严格byte oracle；完整PD提供功能、长度、事件与性能验证，没有固定batch完整logits质量对比。",
            "",
            "## 原始证据",
            "",
            "- [完整比较JSON](artifacts/validation/comparison.json)："
            "原始精度、四组两两比较、逐请求server TPOT、MTP计数。",
            "- [实现与NPU验证](README.md)、[独立review](review.md)、[特性提交](artifacts/code/commit.txt)。",
        ]
    )
    for label, run in zip(labels, runs):
        lines.append(f"- [{label}：{run['run']}](artifacts/runs/{run['run']}/bench/results.json)。")
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
