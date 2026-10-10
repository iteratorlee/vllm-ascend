# C03：D 侧 LI key / scale 缓存写入融合

日期：2026-10-10。状态：算子验证和完整 PD A/B 已完成；仅实验实例开启。

## 实验范围

保持当前 C02 on 的 D 配置，将 LI 的 INT8 key 与 FP16 scale 两次原生 scatter 合并为一次 Triton launch。只在 `kv_consumer`、LI C8、支持的 tensor 布局下启用；开关 `enable_fused_indexer_cache_write` 默认关闭。主 MLA KV 写入、投影、RoPE、量化、索引选择和 MTP 配置不变。

- P：910c-181/182，PP2 / TP16，维持已有服务。
- D：910c-183/184，DP2 / TP16 / EP32；C02 投影复用保持开启。
- 负载：131072 输入 token、1024 输出 token；64 请求、客户端并发 16；逐 D 使用独立 cache salt。
- 当前 KV 容量：每实例 286976 token；本负载实际峰值每实例 2 条、合计 4 条。graph 最大 192 query token 不作为驻留请求并发。
- MTP 5，draft eager；target `FULL_DECODE_ONLY`。

本实验不依赖 C01 的 block-group metadata。直接读取既有物理 token slot，保留 block writer 优先级与不支持布局的原生回退。两张缓存及主缓存写完后仍使用原来的 PD 通知位置。

## 验证与数据

| 验证 | 当前结果 |
| --- | --- |
| 910C byte oracle、动态 ACL graph、边界及回退 | 最终 32 项通过 |
| config / SFA / 新 writer 相关 UT | 最终代码 100 项通过 |
| Ruff 0.14.0 | check / format 通过 |
| 全量 UT 尝试 | 158 通过、2 跳过；在未修改的 A2 `update_graph_params` 用例处失败，与 C02 相同 |
| 完整 PD A/B | A2/A3/B0/B1 各 64/64 成功；详见 [结果](results.md) |

局部微基准使用真实缓存容量 286976，20 轮交替顺序，每轮每实现 200 次 ACL graph replay。默认每 program 1 行。

| 写入行数 | 两次 native scatter 平均 µs | 融合平均 µs |
| --- | ---: | ---: |
| 2 | 17.650220 | 8.114535 |
| 12 | 23.010265 | 9.293900 |
| 16（12 有效、4 PAD） | 24.902585 | 8.187515 |
| 192 | 101.983779 | 20.253655 |

这些数字仅表示写入子路径，不能推算完整 ITL 或 TPOT 收益。

## 正确性契约与发现

1. key 宽度 128、INT8；scale 宽度 1、FP16；slot 为 INT32/INT64；同一 NPU device。
2. 有效 slot 必须唯一，沿用调度器契约。负数、PAD、越界 slot 跳过；地址运算使用 INT64，先替换无效 slot 再计算地址。
3. key / scale / slot 行数及两张缓存容量须相同；不匹配在任何写入前抛错。空批不 launch。
4. 支持连续列和独立行 stride，目标行必须非重叠。实测安装版 Ascend Triton 对 INT8 列 stride=2 无法保持字节一致，已明确回退；不能称为任意 stride 支持。
5. 安装版 CANN 拒绝非连续 inplace 输出。行 padding 的 native oracle 在密集逻辑 view 上 scatter 后复制回去；融合实现直接写原来的行 stride。连续布局使用原生直接 oracle。
6. FP16 任意位模式、未写入位置、行间 padding 与动态 graph 的 valid↔PAD、INT64 极值均按底层字节比较。

支持范围是当前 ND 的独立输入/缓存，不包括源和目标发生写入 alias 或任意 NPU storage format。已核对当前 runner 的 raw buffer 分配与 dtype/shape view，LI cache 未做 NZ 转换；SFA dynamic quant 的 key 与持久 cache 独立，scale cast 为 FP16。32 项用例不代表穷举全部 65536 种 FP16 位模式。

`msprof --pid` 对 worker namespace PID 与映射 host PID 均拒绝，未获得完整服务 kernel trace。失败日志保留，相关练习请求不作为正式 A/B 数据。

## 文件导航

- [review.md](review.md)：独立 expert 审查。
- [results.md](results.md)：正式 A/B 的分轮 TPOT、ITL 和 MTP 接受率；全请求 ITL 改善，饱和 server p99 与 TPOT 未一致改善。
- `artifacts/`：原始正确性、性能、启动、请求和指标证据。
- [本地实验脚本](../../../../deployment/experiments/glm53-migration/C03/)：重启、等待与测试、冻结 trace、分析。

远端根目录：`/public-flash/liyan/glm52-vllm-ascend/experiments/glm53-migration/C03`。`vllm-ascend/` 为隔离分支；`scripts/` 为独立启动与验证脚本；`validation/` 为算子测试；`code/` 为 commit/patch；正式请求数据位于共享根目录 `runs/20261010-c03-*`。

代码提交：`429864a070b2684848e2c5be729a6f9162dc5db1`；author / committer / sign-off 均为 `liyan <yan.l@pku.edu.cn>`。仅核心三文件的增量用于运行环境，保留既有 NATIVE 修复。开关改变后重启并重新捕获 target graph。

前两轮 `control-a0/a1` 保留作开发阶段数据；主比较使用全部独立 NPU 验证结束后运行的 `control-a2/a3`，期间不执行其他算子测试、microbenchmark 或 profiler。B 侧预先完成同负载功能 probe，再运行两轮正式比较。
