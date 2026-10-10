# Profiling 后的优化顺序

本轮 C03 实现和正式 A/B 已完成，随后取得 P/D 的 64 rank profiling。本文根据[实际报告](../../../../glm52-1p1d-vllm-ascend-profiling.md)调整后续顺序；C11 和以下 C14 均未实现，未测得它们的性能收益。

## 1. 优先定位 C14：RoPE 表视图的重复物化

D 的 `aclnnIndex_SliceAiCore_Slice` 共 6528 次，累计 2266.902 ms，占 NPU 任务累计时间 6.451%。真实输入为 `1048576,128`，输出为 `1048576,64` BF16；每次逻辑输出 128 MiB。两个 D TP0 各有 204 次，均为非 graph 任务。累计时间和逻辑字节数均不能换算成可回收的 ITL 或实际 DRAM 流量。

**源码线索，尚未确诊 owner：** 四个 live role 的 `ops/rotary_embedding.py` SHA256 相同，已保存[实际文件](../../../profiling/20261010/artifacts/code/rotary_embedding-live.py)。`_record_cos_and_sin_cache_interleaved` 先 repeat 再 chunk/squeeze，保留 cos/sin 半宽 strided view；`get_cos_and_sin_mla` 对视图执行 `[positions]`。模型为 default RoPE、interleave=true、最大位置 1048576。该路径与大表 Slice 的尺寸相符；此次未开启 stack/module，不能仅凭相邻 CPU 事件认定所有 Slice 都来自它。

### 最小验证

1. 在独立 NPU probe 中执行实际 helper，记录 cos/sin 的 shape、stride 和 storage，并对照 trace 的 CPU/NPU connection。先复现相同大表 Slice。
2. 单独比较两种等价数据搬运：初始化时把只读 cos/sin 视图各自变为 contiguous；或由支持源 stride 的 gather 只取 positions 对应的少量行。先保留当前 repeat/interleave 的精确布局和 dtype。
3. 使用逐字节 oracle，覆盖 positions 的重复、乱序、首尾及 128k 边界，行数 0/1/2/12/192，连续多次更新。不得通过重新计算 trig、改变 RoPE 数学或取消 BF16 舍入取得表面加速。
4. 校验初始化峰值和常驻 HBM、原 backing storage 的所有权、target graph buffer 地址及 replay；确认 KV 初始化容量仍为 286976 token/实例。
5. 若 probe 成立，再以默认关闭的独立配置接入当前 D。做关闭 profiler 的新 A/B，两侧均采用匹配的 D 重启/捕获/预热，保持 128k/1k、实际 2+2 并发。完整比较 server/client TPOT/ITL 的 p50/p75/p90/p99 与 MTP 接受率；不能沿用此前长时间运行的 A 当作严格配对控制。

此候选来自当前 GLM-5.2 profiling，并非直接搬入 SGLang/FlagGems 的模型专用算子。它优先于继续压缩 C03 writer：后者本次只占 NPU 累计时间 0.154%，但这也不构成 C14 收益预测。

## 2. C11：接通 MTP index reuse

两个 D TP0 都观察到 17 次 target graph 执行，每个本地 graph 边界窗口有 5 次非 graph LI/SFA/writer。它与当前 MTP5 eager 路径一致，支持继续验证[完整 C11 设计](c11-design.md)。

后续必须同时验证 wrapper→impl 转发、step0 重置、sampled-row compaction、别名 buffer 去重和 step1+ 索引复用。新 key 的预处理及 writer 仍应逐步执行。仅接通 `skip_topk`、忽略索引行映射，可能改变输出和接受率。

## 3. 其余顺序与证据边界

| 候选 | 当前证据 | 下一步 |
| --- | --- | --- |
| C13 query/RoPE/quant 融合 | D LI 与预处理仍有可见时间；具体细链成本需按真实 shape 划分 | 在 C14/C11 独立验证后再做数值 oracle 与 launch 分析 |
| C09 draft graph | target 有 graph；draft 的非 graph 设备执行已核实 | 先补逐 step metadata、地址和 padding 生命周期，再验证捕获 |
| P SFA/LI | 两者占 P NPU 累计时间 47.719% | 优先研究当前长 prefill 算子/工作空间；本次只有单个冷请求，需补充稳态重复样本 |
| P metadata/通信 | 原榜含长寿命 AI_CPU 与重复 HCCL envelope | 先关联传输和调度，不能按原始累计时间排名直接削减等待 |

每个候选保持独立开关、独立 commit、独立 A/B；代码 author/committer/sign-off 使用 `liyan <yan.l@pku.edu.cn>`。
