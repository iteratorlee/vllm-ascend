# C03：D 侧 LI key / scale 融合写入独立审查

## 结论与范围

审查日期：2026-10-10。**在当前 D 的 INT8 `[tokens, 128]` key、FP16 `[tokens, 1]` scale、非重叠目标缓存和有效 slot 唯一的条件下，未发现阻断受控 PD 验证的核心正确性问题。** 本审查发现的目标行重叠问题已增加静态回退 gate；补审已核对修订源码及最终 32 项 NPU 测试日志，R1 关闭。完整 PD 集成及性能结果仍待后续验证。

本轮仅阅读源码、测试、微基准和目标仓库 `AGENTS.md`，仅写本文件，没有修改实现、执行 NPU 测试或操作服务。当前环境为 C02 已开启、draft `enforce_eager=true`；不能把独立 writer 的 ACLGraph 测试称为完整 MTP graph 验收。

C03 根据目标 SFA 源码中的两次 `npu_scatter_nd_update_` 实现一次 Triton launch 双写。它使用原有 **token slot**，不使用 C01 的 group metadata，没有迁移 C01 的分组容量假设，也没有修改 key 量化、scale 数值、LI top-k 或 attention 算法。

审查工作目录：`/private/tmp/c03-development/vllm-ascend`。核心配置和接入 diff 已与 [C02 归档 SFA](../C02/artifacts/code/source/vllm_ascend/attention/sfa_v1.py)、[C02 归档配置](../C02/artifacts/code/source/vllm_ascend/ascend_config.py)核对：新增配置、D 角色门控和共同写缓存位置的分支构成 C03 核心增量。主执行者报告开发基于 C02 后续工作基线 `eb246406dcf7456eb33dd602181761f6fb311e4d`；补审已读取[最终提交记录](artifacts/code/commit.txt)，C03 commit 为 `429864a070b2684848e2c5be729a6f9162dc5db1`，author、committer 和 signoff 均为用户指定的 `liyan <yan.l@pku.edu.cn>`。

## 实现核查

以下行号对应补审后的本地文件。

| 核查项 | 源码位置 | 审查结论 |
| --- | --- | --- |
| 配置与角色 | `ascend_config.py:294`；`sfa_v1.py:586` | `enable_fused_indexer_cache_write` 默认为 `False`，只接受 `bool`，与 sparse 模型及 `is_kv_consumer` 相与。准确范围是 consumer 实例；没有额外把 attention state 限定为 DecodeOnly。P-only 和无 KV transfer 的实例不启用。 |
| dtype / 形状门控 | `fused_indexer_cache_write.py:60` | 必须同一 NPU，key/cache 都是 INT8、scale/cache 都是 FP16，slot 为 INT32/INT64；仅接受二维 128 列 key、一列 scale、一维 slots。A5 的 FP8/FP32 LI 格式回退，不会被当作此 INT8 路径。 |
| 不支持布局回退 | `fused_indexer_cache_write.py:80` | key 源和目标的 column stride 都必须为 1，覆盖主执行者已发现的 Ascend INT8 column-stride backend 问题。源、目标和 slot 的 row stride 分别传入，允许独立行 padding；目标 key/scale row stride 分别至少为 128/1，拒绝重叠行。未使用强制 contiguous 复制。 |
| 行数与容量 | `fused_indexer_cache_write.py:88` | key、scale、slot 行数必须相等，两个缓存容量必须相等；不通过取最小值吞掉漏写。异常发生在 launch 之前。N=0 在元数据校验后返回成功，无零 grid launch。 |
| int64 寻址 | `fused_indexer_cache_write.py:30` | program/row 和读取后的 slot 在地址乘法前转为 INT64。对无效 slot 使用 `safe_slot=0`，key/scale 的 load 和 store 同时受 `valid` mask 控制。负数、PAD 和超上界不产生实际写入。 |
| 双写一致性 | `fused_indexer_cache_write.py:35` | 同一个 slot 和同一个有效性条件用于 key、scale；不做浮点运算或 dtype 转换。内核内两个 store 不构成跨 stream 的原子事务；正确性依赖原有 stream/event 顺序，不能由其他 stream 无同步地观察中间状态。 |
| SFA 接入点 | `sfa_v1.py:2006` | 在 NATIVE、PROLOG、MLAPO 分支汇合后，用原两次 scatter 相同的 cache view、slot 和 key/scale view 调用。没有改变主 KV 的写入、LI 前处理或查询侧运算。 |
| writer 选择 | `sfa_v1.py:2015` | 新调用要求 `HAS_TRITON`、开关、LI C8 和非空 scale；已有 `_use_li_c8_reshape_optim()` 为真时继续走 block writer。融合成功才跳过原来的两次 scatter；返回 False 时保留原 native 路径。 |
| connector / 消费者顺序 | `sfa_v1.py:2066` | `notify_kv_cache_written` 仍在写缓存分支后，后续 LI 查询仍在共同路径执行。没有把 connector 通知或 LI 读取提前到 writer 前面，也没有新增 stream。 |
| C02 与 MTP 生命周期 | `sfa_v1.py:1887`、`:2023` | C03 不存储输入值、slot 或投影到跨调用状态。每次执行消费当次传入 tensor；C02 的局部 projection 不受改动。MTP 若使用该 consumer 实例，遵循同一调用规则。 |
| graph 重放 | `fused_indexer_cache_write.py:96` | kernel 在每次 replay 读取固定地址中的新 slot/key/scale；编译常量来自 shape/stride，不来自 device 内容。没有新增 `.item()`、host sync、host metadata 或持久 workspace。完整模型捕获是否正确选择和重放此路径仍需 PD 实测。 |

`HAS_TRITON` gate 是本次最终接入补充之一，已实际读取到。它防止在该条件为假时调用新 kernel；新增模块仍在文件顶部导入，方式与已有 `rope` Triton 模块一致。本次未验证一个实际未安装 Triton 的运行环境，也不将静态 gate 等同于该环境的启动验收。

## 独立发现与处理状态

### R1：目标缓存的重叠行回退——已修订并补审

首次审查版本只限制 key column stride=1，仍可接受通过 `expand` 或 `as_strided` 构造的目标：例如 key cache 的 row stride 为 0 或 64，scale cache 的 row stride 为 0。即使有效 token slots 唯一，不同逻辑行也会写入重叠物理地址，不能满足独立行缓存的契约。

当前实际 KV cache 使用正常分配的非重叠行，所以上述问题**不是已观察到的生产缓存损坏，也不构成当前 canonical layout 的阻断项**。但 helper 对外声明支持 layout fallback，因此应把这个限制写入静态 gate。

补审确认 `fused_indexer_cache_write.py:85` 已要求 key cache `stride(0) >= 128`、scale cache `stride(0) >= 1`，不满足时在 launch 前返回 False。新增 2 个 NPU 参数用例分别构造 key `as_strided((16,128),(1,1))` 和 scale `expand(16,1)`，检查 False 及两个底层缓存均未写入。已读取[最终 NPU 日志](artifacts/validation/npu-cache-write-final.log)：`32 passed, 15 warnings in 4.89s`。修订没有改变正常缓存的 kernel 或地址计算，R1 已关闭。

此外，源数据和目标缓存不得发生会被其他 program 写入覆盖的 alias；当前量化输出与持久 cache 独立，符合条件。本审查没有要求加入 host 数据检查或为正常路径引入复制。

### R2：门控和完整调用链仍主要由静态审查支持

新增 UT 检查配置类型、缺省、非 sparse 关闭及 CPU 不写入回退；新增 NPU 用例直接调用 helper。它们没有执行完整 SFA forward 来验证 P/D 角色、`HAS_TRITON=False`、已有 block writer 优先、helper 返回 False 后两个 scatter 各执行一次，以及 helper 成功后原两个 scatter 都跳过。

这些接线已逐项静态核对，未发现错误。主执行者计划以真实 PD 开关加载和运行补足集成证据；在取得该证据前，只能称 kernel / graph 测试通过，不能称完整模型集成测试通过。开关在实例初始化和 graph 捕获时确定，切换配置应重启并重新捕获，不能假定修改 Python 属性会改写已有 graph。

## 测试与微基准证据边界

测试由主执行者执行；补审已读取本地原始 NPU、受影响 UT、格式检查、全量 UT 尝试日志和微基准 JSON，没有独立重跑。远端归档目录为 `/public-flash/liyan/glm52-vllm-ascend/experiments/glm53-migration/C03/validation`；下列本地链接固定此次核查材料。

| 验证 | 报告结果 | 从源码确认的实际范围 |
| --- | --- | --- |
| [910C kernel / graph / 边界测试](artifacts/validation/npu-cache-write-final.log) | 32 项通过，15 warnings | 28 个参数组合：N=0/1/2/6/12/16/192，连续或行 padding，INT32/INT64 slot；另有行数/容量错误、column-stride 回退和重叠目标回退共 4 项。N=0 不捕获 graph，其余组合执行变化输入重放。 |
| [受影响 UT](artifacts/validation/affected-ut-final.log) | 100 项通过，15 warnings | 包含本次 8 项新增 UT；不能在这 100 项之外重复累计这 8 项，也不等同于全仓测试通过。 |
| [writer ACLGraph 微基准](artifacts/validation/writer-benchmark.json) | 2 行 17.6502→8.1145 µs；12 行 23.0103→9.2939 µs | 已与 JSON 的 tile 0/1 均值核对，20 轮、每轮 200 次 replay。该测量早于新增 row-stride gate；不应称为最终修订另行重测结果。未包含完整 LI、attention、MoE、跨 rank 通信和 PD 调度。production 使用默认 tile=1。 |

[原始 oracle 失败日志](artifacts/validation/npu-cache-write-initial-oracle-failure.log)确实记录 native 对非连续输出报 `Output tensor should not from workspace`；[column-stride 失败日志](artifacts/validation/npu-column-stride-failure.log)记录 byte mismatch。这些失败及修订历史被保留，没有用最终通过覆盖原始发现。

[格式/UT 汇总](artifacts/validation/final-lint-and-ut.log)记录 HAS_TRITON 条件超行宽，格式化后 `All checks passed`；[最终布局修订检查](artifacts/validation/safety-final.log)也记录格式检查通过，但其中的 `30 passed` 是增加 2 个重叠用例前的执行片段，应以独立 `npu-cache-write-final.log` 的 32 项为最终 NPU 数量。

[全量 UT 尝试](artifacts/validation/full-ut-attempt.log)为 **1 failed、158 passed、2 skipped** 后停止；失败是 `test_update_graph_params_handles_captured_paged_attention_params` 期望 `graph_task_update_begin` 被调用一次、实测 mock 调用为零。该测试位于普通 attention graph 更新路径，不是新 writer 用例；本审查没有独立基线重现，不能把它直接归因于 C03，也不能宣称它已证实与 C03 无关。全量 UT 未通过这一边界必须保留。

### 正确性 oracle 的有效性

- oracle 先把 native 不接受的非连续输出 view 转为 dense logical view，执行原两次 native scatter，再 `copy_` 回 oracle 的原 view；融合侧直接在真实 strided view 上写入。这比较的是逻辑 scatter 语义，不宣称该 CANN build 可原地 scatter 到非连续输出。
- 比较对象是整个脏 cache 的底层 key/scale tensor，转 `uint8` 后逐 byte 比较，覆盖未写行和行间 padding，强于仅比较写入位置或浮点近似相等。
- graph 连续 5 次改写 slot、key、scale，再与当次 native 结果比较。覆盖有效末行、负 slot、上界 slot、全 PAD，以及 INT32/INT64 极值，能够检测把捕获期 slot/数据当成常量的错误。
- scale 由随机 INT16 bit pattern reinterpret 为 FP16，避免只用普通正浮点数的狭窄分布；这不是穷举全部 65,536 种 bit pattern，也不保证每个用例出现 signed zero、NaN 和 Inf。
- row-padding case 使用 key row stride=256、scale row stride=2、slot stride=2；测试未穷举任意 padding、非零 storage offset、所有 alias 关系及所有设备。生产所用 contiguous payload、正常非重叠行与这些范围相符。
- 测试按调度契约保持**有效 slot 唯一**，invalid slot 可以重复。没有验证重复有效 slot 的确定性，也不应为此假定多个 program 的写入顺序。当前 helper 把这一约束写入 docstring，没有在热路径做 device→host unique 检查。

### 性能解释

`bench_writer.py` 为每种 tile 捕获图，先比较完整输出，再使用 NPU event，交替正序/倒序测量多轮并保留 paired samples。基准包含 2、12、16（12 个有效 slot）、192 行，以及真实量级的 token-row cache 容量和跨 128k/256k 附近 slot。

该微基准在同一组地址反复 replay，不能代表完整模型的数据局部性、并发干扰或跨 rank 尾延迟；NPU event 的区间均值/分位数也不是逐 token 的 ITL p99。现有结果足以支持继续受控 PD 验证，**不能据此给出 TPOT、ITL p99 或吞吐改善预测，也不能按模型层数直接外推微秒收益。**

## 部署与验收边界

1. 应只部署 C03 增量，并保留 live D 已有 NATIVE 修复、C02 及先前运行补丁。不能用干净开发树的完整 SFA 文件覆盖 live 文件而无意撤销旧改动。
2. PD 验证应记录新开关确实加载、实际 writer 路径、正常 token 输出/请求结束、MTP acceptance 和错误日志，再比较相同负载下的 ITL/TPOT。当前 32 项 kernel 用例未覆盖 connector 同步、完整模型图、实际多 rank 消费者或全量请求生命周期。
3. 当前 D draft 仍为 eager。C03 本身不解除 MTP graph gate，不提供 C09 常驻 metadata，也不代替 C11/C12 的独立正确性工作。
4. 回退语义是继续原有实现；对于原 native 本就不支持的布局，返回 False 不会创造新的 native 支持能力。默认关闭仍是清晰的功能回退途径。

已按目标 `AGENTS.md` 检查：无新环境变量、可变全局状态、host sync、upstream monkey patch 或 model runner 改动；新增常量为不可变宽度。新增功能提供 UT、真实 NPU 子路径测试和微基准。完整测试、格式检查及最终提交状态应由主执行者按原始日志记录，本审查不宣称全仓测试或提交门槛已全部通过。

## 本次读取的文件指纹

以下 SHA-256 固定补审输入，**已包含 R1 gate 和新增 2 个回退 case**。除 kernel 和 NPU 测试外，其余指纹与首次审查相同。

| 文件（相对于审查目录） | SHA-256 |
| --- | --- |
| `AGENTS.md` | `dc198b13b3dea93f57883b2c7bedb44ad693b83f5e7e74a3b7f2b558eceae97a` |
| `vllm_ascend/ops/triton/fused_indexer_cache_write.py` | `3c4e06b88756eeabec9047e9518b2eeda6e7cb83f84237f365f6775bf85b7017` |
| `vllm_ascend/ascend_config.py` | `a3458b6e559f693b520e371cdb21d54a72ab470d50635a35630cb3e4aede775f` |
| `vllm_ascend/attention/sfa_v1.py` | `7371fc2840f0147f0b851dd4544cb635dc959f7b569926bee58691513068518a` |
| `tests/e2e/nightly/test_fused_indexer_cache_write.py` | `1de2000be28e5a4e693737bb12c9ada7e5101a8920dc02016a7807125be2602c` |
| `tests/ut/attention/test_fused_indexer_cache_write.py` | `14b5ab230d704b47eafdc03aa36e79cd6e82e76927e271d37ab9ec83a2292120` |
| `tools/glm52_c03/bench_writer.py` | `b68bbf4e5983b17a9358e57a558446d02d155e0015e06a661aa6c5a13bc41139` |
