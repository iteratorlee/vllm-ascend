#!/usr/bin/env python3
"""Render the root report from auditable local profiling summaries."""
# Preserve the Markdown template's table rows and prose paragraphs verbatim.
# ruff: noqa: E501

import argparse
import json
import textwrap
from pathlib import Path


def main() -> None:
    HERE = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, default=HERE.parent)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path, default=Path.cwd() / "glm52-1p1d-vllm-ascend-profiling.md")
    parser.add_argument("--base", default="docs/profiling/20261010")
    args = parser.parse_args()
    BASE = args.base
    analysis_dir = args.analysis or args.profile_dir / "analysis"
    artifacts_dir = args.artifacts or (
        args.profile_dir / "artifacts" if (args.profile_dir / "artifacts").is_dir() else args.profile_dir
    )
    data = json.loads((analysis_dir / "kernel-summary.json").read_text())["roles"]
    traces = json.loads((analysis_dir / "trace-evidence.json").read_text())
    quantile_records = json.loads((artifacts_dir / "validation/kernel-duration-quantiles.json").read_text())["records"]
    quantiles = {(q["role"], q["name"], q["type"]): q for q in quantile_records}
    parts = []

    def add(text: str) -> None:
        parts.append(textwrap.dedent(text).strip())

    def table(headers: list[str], rows: list[list[object]]) -> str:
        return "\n".join(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
            + ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows]
        )

    def fmt(value: float) -> str:
        return f"{value:,.3f}"

    add(f"""
    # GLM-5.2 910C vLLM-Ascend PD profiling（2026-10-10）

    本报告使用当前实验实例的真实采集结果。P、D 各 32 份 `kernel_details.csv` 全部参与统计；本地另有四份 TP0 代表 `trace_view.json` 供路径核对；本文使用 D 两份做 graph / 非 graph 调用计数。**Top20 来自 64 份全 rank CSV，不是只取代表 rank。**

    阅读导航：[P Top20](#41-p-的-top20) · [D Top20](#42-d-的-top20) · [通信与 rank](#5-通信长寿命任务重叠与-rank-差异) · [MTP/C03 路径](#6-c03mtp-与大表-slice-的实际路径证据) · [复现与原始数据](#7-复现精度与证据导航)。

    文件名 `glm52-1p1d-vllm-ascend-profiling.md` 按任务约定保留。实际部署是一个逻辑 P（PP2×TP16）和两个 D 数据并行实例（DP2×TP16、EP32）；不能把“1p1d”理解为各一台机器或各一个 rank。

    ## 1. 已观察到的热点与边界

    - **P 的主要 NPU 热点是稀疏 attention 与 LI。** `SparseFlashAttention` 占 NPU 任务累计时长的 37.972%，`LightningIndexerQuant` 占 9.747%，合计 47.719%。MoE grouped matmul 和量化 matmul 也较重；原始总榜中的 AI_CPU 元数据任务与通信区间需要另读。
    - **D 的工作分散在 attention、量化 matmul、MoE 分发/合并和预处理。** NPU 表中 SFA 为 11.578%，NZ QuantMatmul 为 10.584%；`MoeDistributeDispatchV2` + `MoeDistributeCombineV2` 为 16.145%，这两类 NPU kernel 包含通信语义，不能当作纯矩阵计算。
    - **D 存在可进一步定位的大表 Slice。** `aclnnIndex_SliceAiCore_Slice` 为 6,528 次、累计 2,266.902 ms、平均 347.258 μs，占 NPU 累计时长 6.451%；真实输入 `1048576,128`、输出 `1048576,64`、BF16。它的优先级高于继续仅缩短已经很小的 C03 writer。
    - **C03 writer 确实执行，MTP 路径 LI 重算也有设备执行证据。** D 全 rank `_fused_indexer_cache_write` 执行 14,144 次、累计 54.284 ms；两个 D TP0 trace 各有 17 次 target graph 执行，每轮之后恰有 5 次非 graph LI/SFA/writer，符合 MTP5 eager 的运行路径。详见第 6 节；这不等于 C11 已实现或已取得收益。

    这些比例是任务 `Duration` 累计值的构成，**不是墙钟占比、硬件利用率或可以直接回收的 ITL 收益**。正式 C03 性能判断仍以 [完整 PD A/B 结果](docs/potential-opt-from-glm53/experiments/C03/results.md) 为准；本次只有 C02/C03 on 的诊断采样，不构成新的 on/off 性能对照。

    ## 2. 部署、负载与采集完整性

    | 项目 | P | D |
    | --- | --- | --- |
    | 节点 / 角色 | 172.16.10.181、182；一个 producer | 172.16.10.183、184；两个 consumer 实例 |
    | 并行 | DP1、PP2、TP16；层分配 `41,37` | DP2、PP1、TP16、EP32；DCP1 |
    | 模型 | `/public-flash/models/GLM-5.2-w8a8` | 同一 target；MTP loader view |
    | 图与草稿 | 本次单次 prefill 请求 | target `FULL_DECODE_ONLY`；MTP5 `enforce_eager=true` |
    | 输入 / 输出 | 131,072 / 1 token | 每请求 131,072 / 1,024 token |
    | 采样负载 | 新 prefix salt；一条完整冷 prefill | 16 请求、client concurrency 16；8 请求/decoder；独立 D cache salt |
    | 采样期间并发 | 不适用 | `/start_profile` 前、启动后与 `/stop_profile` 前均观测 Running=2+2；窗口末 Waiting=6+6 |
    | 导出范围 | 32 CSV，PP0/PP1 各 16 TP rank | 32 CSV，DP0/DP1 各 16 TP rank |

    版本与代码状态：四台使用 `quay.io/ascend/vllm-ascend:v0.26.0rc1-a3`。环境盘点记录 vLLM `0.26.0+empty`、vLLM-Ascend `0.26.0rc1`、PyTorch `2.10.0+cpu`、torch-npu `2.10.0.post4`、Transformers `5.14.1`，CANN `9.1.0`；见 [环境盘点](docs/01-environment/inventory.md)。此次服务安装代码以原 repo HEAD `f2f74a16c3c50a76f4349d807918e83edec1e35c` 为基线，再叠加 C02/C03、NATIVE 与 record_shapes runtime 补丁。隔离实验 repo 的 feature commit 不等于服务安装 HEAD。此次各 role 的版本、安装 HEAD、关键文件 SHA 与镜像状态见 [runtime/image 证据目录](docs/profiling/20261010/artifacts/code/)，应结合文件 SHA 与 patch 判断实际生效代码。

    P 的 prefix counter 增量为 hits=0、queries=131072，支持完整冷前缀路径。D 16 请求全部 HTTP 200，实际 usage 均为 131072/1024，且采样前后没有 preemption 增量。**客户端 16 并发和 12 个 padded query 行均不能代替实际驻留并发 2+2。**

    控制记录中 P 请求墙钟为 7.510 s；D 启动 profile 后手动约 1 s 停止。导出设备任务区间约为 P 单 rank 5.109–6.606 s、D 单 rank 1.193–1.212 s；它们与 HTTP 控制区间不是同一个时钟或同一个边界。`stop_profile` 返回含导出/收尾开销，不能记作推理时长。

    采集与离线解析说明：P/D start/stop 均成功；worker daemon 拒绝自动 parse 后，离线 `analyse` 成功导出全部 64 表。启动参数写了 `torch_profiler_record_shapes=true`；原 wrapper 缺少参数转发的问题已在 C03 正式 A/B 之后、此次 profiling 采集之前修复，四个 live role 均应用后重启采集，17 个 UT 通过。证据为 [record-shapes runtime patch](docs/profiling/20261010/artifacts/code/record-shapes-runtime.patch) 和各 role 的 `profiling-patch-*.log`。**此次 trace 带开启 shape 的修复，但本报告仍只采用实际导出字段。** N/A 保留，不能据模型配置补造 shape、有效序列长度或模块归属。

    证据：[workload manifest]({BASE}/artifacts/workload/manifest.json)、[控制记录]({BASE}/artifacts/workload/control.json)、[D load]({BASE}/artifacts/workload/d-load.json)、[D 请求]({BASE}/artifacts/workload/d-requests.json)、[P 启动脚本]({BASE}/artifacts/config/prefill-0.sh)、[D 启动脚本]({BASE}/artifacts/config/decode-0.sh)、[采集配置与补丁]({BASE}/artifacts/config/)。

    ## 3. 统计口径

    按 CSV 的 **`(Name, Type)`** 分组，以 `Σ Duration(us)` 降序排序。`AivKernel` 在不同 collective 下类型不同，必须保留 Type；两个不同实际 Name 即使 Type 一样也不合并。累计时长单位 ms，均值单位 μs，次数是导出任务记录数，不等同于模型层、算法调用或输出 token 数。显示三位小数，原始精度保存在 JSON 中。

    原始 Top20 分母是该角色全部 CSV 记录的 `ΣDuration`，保留通信 envelope 与 AI_CPU 记录以便原表复查。NPU Top20 仅包含 `AI_CORE / AI_VECTOR_CORE / MIX_AIC / MIX_AIV`，分母仅为这些 core 记录的累计 `Duration`，排除 `COMMUNICATION` 和 `AI_CPU`；MoE dispatch/combine 仍在 NPU 表中并单独注明通信语义。它是设备计算/搬运热点视图，不是 FLOPs 分类。

    `Wait Time` 与 `Duration` 分开汇总，没有相加。HCCL envelope 可以覆盖等待和内部任务，多个 stream 可以重叠，32 rank 的任务也会并行；因此不能把累计值除以请求墙钟求利用率。区间并集只在**同一 rank** 内计算：`union([Start, Start+Duration])`。跨主机 clock 未校准，不合并绝对时间推导关键路径或 PP/DP 等待。

    输入 shape 的分号分隔各输入，逗号分隔维度，连续分号保留空槽；“常见 shape”按次数取众数，并给出该 shape 的出现次数。它描述传入 tensor 的物理尺寸，不能推断本次有效 token、KV 全部有效 block、量化权重的逻辑布局。尤其 P routing 后 grouped matmul 的众数只出现 7 次，应视为实际示例而非统一 shape。
    """)

    add(
        table(
            [
                "角色",
                "CSV 数",
                "任务行数",
                "Name+Type 组数",
                "原始 ΣDuration ms",
                "NPU ΣDuration ms",
                "有输入 shape 的行数",
            ],
            [
                [
                    r.upper(),
                    x["file_count"],
                    f"{x['rows']:,}",
                    f"{x['name_type_groups']:,}",
                    fmt(x["duration_ms"]),
                    fmt(x["npu_duration_ms"]),
                    f"{x['shape_rows']:,} / {x['rows']:,} ({100 * x['shape_rows'] / x['rows']:.2f}%)",
                ]
                for r, x in data.items()
            ],
        )
    )

    for role in ("p", "d"):
        x = data[role]
        add(
            f"## 4.{1 if role == 'p' else 2}. {role.upper()} 的 Top20\n\n### 原始导出记录 Top20\n\n占比的分母为 {fmt(x['duration_ms'])} ms；本表保留通信 envelope / AI_CPU 原始记录。前 20 项合计 {sum(k['raw_share_pct'] for k in x['top20_raw']):.3f}%。"
        )
        rows = [
            [
                i,
                f"`{k['name']}`",
                f"`{k['type']}`",
                f"{k['count']:,}",
                fmt(k["duration_ms"]),
                fmt(k["mean_us"]),
                fmt(quantiles[(role, k["name"], k["type"])]["p50_us"]),
                fmt(quantiles[(role, k["name"], k["type"])]["p90_us"]),
                f"{k['raw_share_pct']:.3f}%",
            ]
            for i, k in enumerate(x["top20_raw"], 1)
        ]
        add(table(["#", "实际 Name", "实际 Type", "次数", "累计 ms", "均值 μs", "p50 μs", "p90 μs", "原始占比"], rows))
        add(
            f"### NPU 任务 Top20\n\n占比的分母为 {fmt(x['npu_duration_ms'])} ms；前 20 项合计 {sum(k['npu_share_pct'] for k in x['top20_npu']):.3f}%。"
        )
        add(
            table(
                ["#", "实际 Name", "实际 Type", "次数", "累计 ms", "均值 μs", "p50 μs", "p90 μs", "NPU 占比"],
                [
                    [
                        i,
                        f"`{k['name']}`",
                        f"`{k['type']}`",
                        f"{k['count']:,}",
                        fmt(k["duration_ms"]),
                        fmt(k["mean_us"]),
                        fmt(quantiles[(role, k["name"], k["type"])]["p50_us"]),
                        fmt(quantiles[(role, k["name"], k["type"])]["p90_us"]),
                        f"{k['npu_share_pct']:.3f}%",
                    ]
                    for i, k in enumerate(x["top20_npu"], 1)
                ],
            )
        )
        add(
            "### 对应真实输入 shape\n\n原始及 NPU 两个 Top20 的并集如下。相同名称只列一次；更多 shape、数据类型与输出 shape 见汇总 JSON。"
        )
        seen = set()
        shape_rows = []
        for k in x["top20_raw"] + x["top20_npu"]:
            key = (k["name"], k["type"])
            if key in seen:
                continue
            seen.add(key)
            common = k["input_shapes"][0]
            shape_rows.append(
                [
                    f"`{k['name']}` / `{k['type']}`",
                    f"`{common['shape']}`",
                    f"{common['count']:,} / {k['count']:,}",
                    k["input_shape_count"],
                ]
            )
        add(table(["Name / Type", "真实常见输入 shape", "该 shape 次数 / 总次数", "不同输入 shape 数"], shape_rows))

    add("""
    ## 5. 通信、长寿命任务、重叠与 rank 差异

    ### 通信记录重叠

    HCCL `hcom_*` envelope 与内部 `AivKernel` 存在完全相同 `(Type, Start, Duration)`。P 有 44,928 条 envelope 精确匹配设备通信任务，重复部分累计 32,973.896 ms；D 的 96,832 条 envelope 全部精确匹配，重复部分累计 4,301.868 ms。因此原始总量不是可相加的设备工作。

    P 另有 AI_CPU collective 和未与 `AivKernel` 精确匹配的 envelope，不能把“只删一个名字”理解为完整通信成本去重。分析脚本单列所有类别，并以单 rank 区间并集检查覆盖。D 若仅去掉 envelope，记录累计值为 39,443.065 ms，其中独立 HCCL 设备任务 4,301.868 ms；MoE dispatch/combine 的 5,673.474 ms 仍另含通信与 token 搬运语义，不能据此拆出纯链路带宽或网络等待。
    """)
    labels = {
        "npu_other": "其余 NPU 计算/搬运",
        "npu_moe_dispatch_combine": "NPU MoE dispatch/combine（含通信）",
        "hccl_device_task": "HCCL 设备通信任务",
        "hccl_envelope": "HCCL 操作 envelope（可与内部重复）",
        "aicpu_kv_metadata": "AI_CPU KV 元数据",
        "aicpu_communication": "AI_CPU collective",
        "aicpu_init": "AI_CPU HCCL init",
        "aicpu_other": "其余 AI_CPU",
    }
    add(
        table(
            ["角色", "类别", "次数", "ΣDuration ms", "原始占比", "ΣWait ms（另列）"],
            [
                [
                    r.upper(),
                    labels.get(cat, cat),
                    f"{v['count']:,}",
                    fmt(v["duration_ms"]),
                    f"{v['raw_share_pct']:.3f}%",
                    fmt(v["wait_ms"]),
                ]
                for r, x in data.items()
                for cat, v in x["categories"].items()
            ],
        )
    )
    add("""
    ### P 元数据与 init

    `StoreKvBlockMetadata` 416 次累计 24,536.845 ms，进入原始榜第 3，平均 58,982.800 μs。但其中只有 **32 次超过 100 ms**，累计 24,408.128 ms；长任务范围 685.194–840.460 ms，而全部调用中位数只有 237.495 μs。它是长寿命 AI_CPU 元数据任务，不能把其长 elapsed duration 当作 AI_CORE 持续计算或可直接节省的 prefill 时间；需要结合 KV transfer 与调度事件再确定等待比例。

    `hcomAicpuInit` P 每 rank 一次，共 32 次，执行累计仅 114.495 ms；其另列 Wait 累计 7,539.508 ms。执行时长只占原始总量 0.056%，并非稳态 NPU 算子热点；Wait 也不能直接跨 rank 相加为可优化墙钟。D CSV 中没有 AI_CPU/init 记录。P 的若干低次数 HCCL envelope 排在原始 Top20，且单次可达数十 ms，应当先定位启动/同步/等待状态，不能用这样的样本概括稳态传输。

    ### 单 rank 区间并集

    以下仅在各自 rank 的 clock 上计算。NPU 覆盖率 = NPU 区间并集 / 首末导出任务跨度；“全部任务并集”包括通信与 AI_CPU，但已通过区间并集自然避免时间重叠重复。该覆盖率不代表 core occupancy、有效 MAC 吞吐或硬件利用率。
    """)
    representatives = [r for x in data.values() for r in x["ranks"] if r["tp"] == 0]
    add(
        table(
            [
                "角色 / rank",
                "导出跨度 ms",
                "原始 ΣDuration ms",
                "NPU ΣDuration ms",
                "NPU 并集 ms",
                "NPU 覆盖率",
                "全部任务并集 ms",
            ],
            [
                [
                    ("P" if r in data["p"]["ranks"] else "D") + " " + r["key"],
                    fmt(r["span_ms"]),
                    fmt(r["sum_duration_ms"]),
                    fmt(sum(v["duration_ms"] for c, v in r["categories"].items() if c.startswith("npu_"))),
                    fmt(r["union_ms"]["npu"]),
                    f"{100 * r['union_ms']['npu'] / r['span_ms']:.2f}%",
                    fmt(r["union_ms"]["all"]),
                ]
                for r in representatives
            ],
        )
    )
    add(
        "D 存在 NPU stream 重叠，因此 NPU 累计时间大于 NPU 区间并集；P 本窗口 NPU 行区间没有重叠，两者相同。两种情况均不能把所有任务累计时间当作墙钟。"
    )
    add(
        "### 同一 PP/DP 组内 TP rank 的分布\n\n每组 16 个 TP rank。CV 为这 16 rank 数值的总体标准差 / 平均值，属于本次诊断样本。"
    )
    add(
        table(
            [
                "角色 / 组",
                "NPU ΣDuration min / mean / max ms",
                "NPU ΣDuration CV",
                "NPU 并集 min / mean / max ms",
                "并集 CV",
                "HCCL 设备并集 CV",
            ],
            [
                [
                    r.upper() + " " + g,
                    " / ".join(fmt(v["npu_sum_ms"][q]) for q in ("min", "mean", "max")),
                    f"{v['npu_sum_ms']['cv_pct']:.2f}%",
                    " / ".join(fmt(v["npu_union_ms"][q]) for q in ("min", "mean", "max")),
                    f"{v['npu_union_ms']['cv_pct']:.2f}%",
                    f"{v['hccl_device_task_union_ms']['cv_pct']:.2f}%",
                ]
                for r, x in data.items()
                for g, v in x["groups"].items()
            ],
        )
    )
    add("""
    P PP0/PP1 分别分配 41/37 个 backbone 层，应分别比较组内 TP。PP0 各 rank SFA=328、LI=96；PP1 各 rank SFA=304、LI=80，组内次数一致。不能把两个 PP stage 的层数、附加路径或采样跨度差异称为 TP 失衡。D 两个 DP 组的平均 NPU 累计时长分别为 1,099.860 / 1,096.465 ms，接近；全部 rank 的 SFA=1,411、LI=442、writer=442。

    D HCCL 设备区间并集的组内 CV 较高（30.41% / 34.53%），但全部任务并集的 CV 仅 0.27% / 0.23%，NPU 并集 CV 为 3.76% / 4.38%。这提示通信时间分布值得检查；仅据累计/并集差异不足以确定哪个 rank 拖慢关键路径，也不足以宣称网络是唯一瓶颈。

    D 的文件后缀 `rank7` 是本地 TP rank，必须连同 `dp/pp/tp/ep` 解读；例如 `dp1/pp0/tp7/ep23` 与 `dp0/pp0/tp7/ep7` 是不同 worker。分析用组合 key，未把两者误合并成一个全局 rank。

    ## 6. C03、MTP 与大表 Slice 的实际路径证据

    ### C03 writer 已执行

    `_fused_indexer_cache_write` D 全 rank 14,144 次，平均 3.838 μs，累计 54.284 ms，仅为 NPU 累计时长的 0.154%。它证明融合分支实际到达设备，不能单凭配置声称开启；但本次没有原生双写的同负载 profiler 对照，不能由这个数值反推 C03 节省了多少时间。

    两个 D TP0 的 device trace 均把 writer 分为 target graph 的 357 次与非 graph stream47 的 85 次。一个细节是：CSV 中此 Triton writer 的 `Model ID` 全为 `4294967295`，而 trace 中 target 部分明确带 `Model Id=26`；其 `OP State` 也与 LI 的 static/dynamic 分布不同。**因此不能只用 CSV `Model ID` 或 `OP State` 区分 target / draft。** `mla_preprocess_0` 的输入 shape 为 N/A，也不能从模型参数补填。

    ### LI 每轮五次的证据与 C11 边界

    两个 DP 的 TP0 均观察到 17 条设备 `MODEL_EXECUTE`；target graph 任务 `Model Id=26`、stream202，非 graph LI/SFA/writer 任务 `Model Id=4294967295`、stream47。以每条 `MODEL_EXECUTE` 的开始到下一条开始为本地窗口，17 个窗口**各有 5 次非 graph LI、5 次 SFA、5 次 writer**。末窗延伸至此 trace 结尾，没有假设跨主机时间同步。
    """)
    add(
        table(
            ["证据 / 单个 D TP0", "DP0", "DP1", "计数解释"],
            [
                ["MODEL_EXECUTE", 17, 17, "target graph 的本次实际执行次数"],
                ["graph SFA", 1326, 1326, "17 × 78；与 target 78 层一致"],
                ["graph LI / writer", "357 / 357", "357 / 357", "17 × 21；writer 由 trace 区分 graph"],
                ["非 graph LI / SFA / writer", "85 / 85 / 85", "85 / 85 / 85", "17 × 5；每个 graph 边界窗口均为 5"],
                ["CPU LI dispatcher", 85, 85, "_C_ascend::npu_lightning_indexer_quant / aclnnLightningIndexerQuant"],
                ["Index Slice", 204, 204, "全部非 graph stream47；整体计数 17 × 12"],
            ],
        )
    )
    add(f"""
    配置 MTP5 eager、单 next-n 层和上述设备/host 计数共同支持：**当前采样中的非 graph MTP 路径每轮执行了五次 LI，未表现为仅 step0 一次 LI。** 这比“dynamic 就是 MTP”的字符串判断更强；但 trace 没有显式 `draft step`/module 标签，不能声称已逐层记录到 wrapper 的 `skip_topk` 值、buffer 绑定或 compaction 顺序。

    全 rank CSV 中 LI 分布为 static 11,424 次、1,415.156 ms，dynamic 2,720 次、199.578 ms，与两个 TP0 的 357+85 计数一致。static/dynamic 在这里仅用于与 trace 交叉核对；不能推广为普遍的 target/draft 分类规则。尤其 “存在 graph”不代表 draft 已 graph 化。

    [C11 设计](docs/potential-opt-from-glm53/continued/20261010/c11-design.md) 中 wrapper forwarding、step0 reset、sampled-row compaction 与 step1+ 复用仍是候选实现和正确性验证范围。后续应记录实际 owner/impl 的状态与每步 LI 调用，同时验证新 key 投影与 cache writer 仍每步执行；不能只删除后四次 LI，也不能把减少某个 kernel 累计时长换算成 p99 收益。

    ### 大表 Slice / RoPE 元数据候选

    该 kernel 的导出数据类型为 `DT_BF16;INT64;INT64`，输入 `1048576,128;2;2`，输出 `1048576,64`。每次输出的逻辑 tensor 体积为 **128 MiB**；每 rank 204 次对应 **25.5 GiB 逻辑输出**。这只是 shape×dtype 的体积，不是实测 DRAM 流量或带宽。

    两个 D TP0 各 204 次均在非 graph stream47，其累计分别为 72.768 / 71.312 ms；全 rank 6,528 次、2,266.902 ms。整体次数与 17×12 一致，但首个 MODEL_EXECUTE 前有 2 次，前 16 个边界窗口各 12 次，尾窗 10 次，因此不声称 17 个窗口都完整采到 12 次。

    本地 host 上下文包含 `aten::index` 对 `1048576,64` BF16 tensor 的操作；[当前 SFA 元数据实现](docs/potential-opt-from-glm53/evidence/current/vllm-ascend/vllm_ascend/attention/sfa_v1.py) 的 `_build` 调用 `get_cos_and_sin_mla(..., use_cache=(draft_index is None))`。二者支持检查 RoPE 表半宽 view 在按位置 gather 前被完整物化的候选方向，**具体源函数归属仍是推断**：此次没有 Python stack/module label，不能认定每一条 Slice 已映射到该函数。

    补充 [live rotary 实现](docs/profiling/20261010/artifacts/code/rotary_embedding-live.py) 中 `_record_cos_and_sin_cache_interleaved` 先 repeat 再 chunk，squeeze 后的缓存可为 strided view；`get_cos_and_sin_mla` 随后对该缓存做 `[positions]`。模型配置 default RoPE、interleave=true、最大 position=1048576、dim64，与本次维度相符；此源码线索仍未通过 connection_id 或微基准证明该 Slice 的 Python owner。

    下一步可用同一负载验证实际 layout、位置索引和所有权，检查一次初始化时把缓存 contiguous 化是否能避免每步隐式物化，或能否先按 positions 取少量行、再取 cos/sin 半宽，或使用支持源 stride 的 gather；需要精确数值和 layout oracle 后再做独立 PD A/B。这个候选不应与 C11 一并归因，也没有已实现收益。

    路径验证原始 trace：[{traces["ranks"][0]["source"].split("/")[2]}]({BASE}/artifacts/{traces["ranks"][0]["source"]})、[{traces["ranks"][1]["source"].split("/")[2]}]({BASE}/artifacts/{traces["ranks"][1]["source"]})；抽取记录见 [trace-evidence.json]({BASE}/analysis/trace-evidence.json)。

    ## 7. 复现、精度与证据导航

    运行以下命令会重新读取 64 CSV、验证它们与 input manifest 的 bytes / SHA256 一致，再生成完整统计与根报告；trace 分析逐份读取两个 D TP0 trace，校验每轮五次非 graph LI/SFA/writer。

    ```sh
    python3 docs/profiling/20261010/analysis/analyze_kernels.py
    python3 docs/profiling/20261010/analysis/analyze_trace.py
    python3 docs/profiling/20261010/analysis/analyze_quantiles.py
    python3 docs/profiling/20261010/analysis/render_report.py
    ```

    `analyze_kernels.py` 将 CSV `Start Time(us)` 的 tab 去掉，时间和 duration 转为整数 ns 后汇总/做区间并集，避免大 epoch 数值的浮点误差。Top20 是完整 `(Name, Type)` 排序的前 20 项；脚本检查角色各 32 文件，按组合 rank key 解析，所有分组/类别次数与总行数一致。对通信 envelope，只以同 rank 的 `(Type, Start, Duration)` 精确匹配计数；没有靠名字做跨 rank 去重。

    上面的 Top20 表已列平均/p50/p90。单次 kernel 的 p50/p75/p90/p99 完整数值见 [kernel-duration-quantiles.json]({BASE}/artifacts/validation/kernel-duration-quantiles.json) 与 [CSV]({BASE}/artifacts/validation/kernel-duration-quantiles.csv)。它们混合本角色全部 rank 和相应实际 Name+Type 下的不同 shape，使用线性插值；仅是单次任务 duration 的诊断分布，不能视为请求 ITL、独立性能重复样本或 step 的端到端分位数。

    | 证据 | 用途 |
    | --- | --- |
    | [P 原始导出目录]({BASE}/artifacts/trace/p/) / [D 原始导出目录]({BASE}/artifacts/trace/d/) | 本文全部排名与真实 shape 的来源 |
    | [input-manifest.json]({BASE}/artifacts/analysis/input-manifest.json) | 738 项实际输入文件的 bytes / SHA256；清单自身另存 |
    | [kernel-summary.json]({BASE}/analysis/kernel-summary.json) | 全部 Name+Type 组、count/总/均值、shape/dtype、分类、每 rank 区间并集与组内分布 |
    | [analyze_kernels.py]({BASE}/analysis/analyze_kernels.py) / [analyze_trace.py]({BASE}/analysis/analyze_trace.py) / [render_report.py]({BASE}/analysis/render_report.py) / [analyze_quantiles.py]({BASE}/analysis/analyze_quantiles.py) | 可复现脚本；只读取本地证据与写派生文件 |
    | [trace-evidence.json]({BASE}/analysis/trace-evidence.json) | D0/D1 TP0 的实际模型/stream 分组、17 窗口与 LI 示例 |
    | [完整 raw 归档]({BASE}/artifacts/archives/full-raw-trace.tar.gz) | 约 28 GB 原始 trace 的压缩存档；分析表不是原始 trace 的替代品 |
    | [C03 正式结果](docs/potential-opt-from-glm53/experiments/C03/results.md) / [C11 设计](docs/potential-opt-from-glm53/continued/20261010/c11-design.md) | 性能结论与后续候选边界 |

    完整 raw 归档本地与远端 SHA256 均已验证一致：`235af5993368c17ba282bfc61b9578c739c76bc9aae0409e9c6e0f28e0aed304`；本地大小 4,745,180,937 bytes，校验记录见 [full-raw-trace-integrity.json](docs/profiling/20261010/artifacts/archives/full-raw-trace-integrity.json)。

    独立复核结果见 [kernel-independent-check.json](docs/profiling/20261010/artifacts/validation/kernel-independent-check.json)：另一轮读取全部 64 CSV、2,682,368 行，逐 Name+Type 的次数、累计 duration 和排序与汇总一致；48 个原始/NPU Top20 并集组的分位数样本数与累计值也交叉核验。

    最终口径限制：本次 P/D 采样长度和负载不同，不能比较两个角色 ΣDuration 的大小来判断资源比例；P 是一次冷 prefill，D 是当前饱和并发下的短窗口；profiler 会扰动运行，采样未提供 C03 off 对照；未测设备利用率、DRAM 字节、跨机 clock 或完整 critical path；未知 shape、模块归属和等待语义均保留为未知。
    """)

    out = args.output
    out.write_text("\n\n".join(parts) + "\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
