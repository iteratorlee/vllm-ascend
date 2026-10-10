# GLM-5.2 P/D profiling：2026-10-10

远端根目录：`/public-flash/liyan/glm52-vllm-ascend/profiling/20261010-c03`。

C03 正式 A/B 完成后单独采集。P PP2/TP16、D DP2/TP16/EP32，C02/C03 实验开启，MTP5 eager，target FULL_DECODE_ONLY。P 131072 输入/1 输出的冷请求；D 16 个 128k/1k 请求，采样窗口首尾 Running=2+2。本目录的耗时不用于替代正式 A/B。

## 目录

| 路径 | 内容 |
| --- | --- |
| `scripts/` | 配置生成、角色重启、等待 ready 和采集工作流 |
| `config/` | 实际启动参数、四机脚本、补丁与启动日志 |
| `log/` | 各角色运行日志 |
| `workload/` | P 冷请求、prefix counter、D 请求和实际并发、profile 控制记录 |
| `trace/p/`、`trace/d/` | 各 32 rank 的完整原始 SDK trace 与离线导出；CSV/JSON/DB |
| `validation/` | UT、采集、离线解析、服务检查与分析校验 |
| `code/` | 代码快照、patch、commit、包版本和关键文件 SHA256 |
| `analysis/` | 可复现分析脚本、kernel 汇总、graph/MTP 证据和报告快照 |
| `staging/` | 输入传输包、完整原始 trace 压缩归档和 SHA256 |

本地用户指定报告：`glm52-1p1d-vllm-ascend-profiling.md`；数据与脚本在本地 `docs/profiling/20261010/`。远端报告快照保留本地文档链接；可点击的本地报告是正式阅读入口。

## 重算

全部为 Python 标准库分析；需在具有完整 CSV 的文件系统上执行：

```sh
PROFILE=/public-flash/liyan/glm52-vllm-ascend/profiling/20261010-c03
python3 "$PROFILE/analysis/analyze_kernels.py" --artifacts "$PROFILE" --output "$PROFILE/analysis/kernel-summary.json"
python3 "$PROFILE/analysis/analyze_trace.py" --artifacts "$PROFILE" --output "$PROFILE/analysis/trace-evidence.json"
python3 "$PROFILE/analysis/analyze_quantiles.py" --artifacts "$PROFILE" --summary "$PROFILE/analysis/kernel-summary.json" --output-dir "$PROFILE/validation"
python3 "$PROFILE/analysis/render_report.py" --profile-dir "$PROFILE" --output "$PROFILE/analysis/glm52-1p1d-vllm-ascend-profiling.md"
```

报告保留全 rank 原始 Top20，同时排除 COMMUNICATION/AI_CPU 给出 NPU Top20。通信 envelope 与实际通信任务可能重复；多 stream/rank 的累计时间不能当作请求墙钟。Shape 仅引用实际字段。

四个 worker 的 SDK 导出全部完成，profiler 已停止；P/D health 与代理实际请求检查通过。
