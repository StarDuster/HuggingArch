# Benchmark CSV schema (frozen)

适用范围：生成 / 上传 measurement CSV（Kernel Bench），或在 `backend/inference/benchmarks.py` 修改 CSV 解析 / 校验代码时。

这份文件是 schema 的**单一事实源**。`backend/inference/benchmarks.py` 是它的唯一执行器；`huggingarch bench validate <csv>` 是 CLI 入口；Kernel Bench 上传走同一校验。

> 备注：`.claude/rules/benchmarks-csv.md`（本地未追踪）是为本地 agent 准备的轻量指针，内容与本文件保持一致。**改 schema 改这里，不要改 `.claude/` 那份**。

## 字段总览

CSV schema 的字段定义、类型规则、词汇表、版本策略全部集中在本文件。**禁止现编现用** —— 在 yaml / CSV / 校验代码里出现的列名 / 类型 / 词表必须和这里一致；要改 schema 就改这份文件并 bump `schema_version`。

## 规则 1 —— 列集合是闭集

CSV 共 **43 列**，分三组：12 个 identity 列、9 个 measurement 列、22 个 provenance 列。

> 这四个数由 `tests/backend/inference/test_benchmarks.py::TestTheSchemaDocsCountsAreTheSchemas`
> 对着 `ALL_COLUMNS` 核。**改之前先看那个测试红不红** —— v13 加了两列而这里没改，文档说 37、
> 代码是 39，这份"唯一真相"和校验器矛盾了一周没人发现，就是因为当时没有这道门。

**Identity 列（12，全部 required）**：

| 列名 | 类型 | 来源 / 约束 |
|---|---|---|
| `model` | str | 模型 ID，例 `deepseek-ai/DeepSeek-V3` |
| `spec_hash` | str | `resolved_spec` 序列化后的 SHA256 前 16 hex；测量时的 spec 版本指纹。变了就是新行（stale-join 拒绝）|
| `phase` | str | `prefill` / `decode` / `unified` 三选一 |
| `parallel` | str | 并行配置，例 `dp=1,ep=2,pp=1,tp=4`；逗号分隔 `key=value`，**key 必须按字典序排列**（validator 强制） |
| `framework` | str | 例 `vllm` / `sglang` / `trt-llm`；纯小写，无版本号 |
| `framework_version` | str | 例 `0.6.3`；语义版本字符串 |
| `gpu` | str | `backend/inference/gpu_specs.csv` 的 GPU SKU 行 ID，例 `H100-SXM` |
| `op` | str | Program leaf 的 primitive op 名（例 `linear`、`flash_attention`、`attention_score`）|
| `node_spec_id` | str | `AnalysisResult.per_op_timing[i]["spec_id"]` —— canonical primary-folded id |
| `shape_json` | str | JSON 字典字符串，shape symbol → 数值，**key 限制见规则 2** |
| `dtype_env_json` | str | JSON 字典字符串，5-slot dtype 绑定，**key/value 限制见规则 3** |
| `kernel_impl` | str | framework_binding 里的 `kernel_impl`，例 `flash_attention_v3` |

**Measurement 列（9，全部 required）**：

| 列名 | 类型 | 来源 |
|---|---|---|
| `latency_us` | float | 平均 latency，微秒 |
| `p50_us` | float | 50th-percentile latency |
| `p95_us` | float | 95th-percentile latency |
| `samples` | int | 用于求 latency 统计的采样次数 |
| `count_per_step` | int | 每 inference step 里该 op 被调用的次数；与 `AnalysisResult.per_op_timing[i]["count_per_step"]` 同义 |
| `achieved_tflops` | float | 实测算力，TFLOPs/s |
| `achieved_bw_gbps` | float | 实测带宽，GB/s |
| `peak_tflops_at_test` | float | 测量时的 GPU peak TFLOPs/s 快照 |
| `peak_bw_at_test` | float | 测量时的 GPU peak HBM 带宽 GB/s 快照 |

**Provenance 列（22）**。这些列有两个不同的性质：

- **必须有值**（4 列）：`schema_version` / `contributor` / `date` / `cross_check`。
- **其余 18 列值可为空**。空各有各的含义，见下表每列自己的说明——空不是"没填"，是那一列定义好的"未观测 / 不适用"。
- **可从 header 缺列**（11 列，`OPTIONAL_COLUMNS`）：`moe_block_member`（v8 起）/ `input_seed`（v9）/ `measured_dtype_env_json`（v10）/ `launch_signature`（v11）/ `outcome`（v12）/ `xcheck_typical` / `xcheck_worst`（v13）/ `kmd_version` / `umd_version`（v14）/ `outcome_detail`（v15）/ `timing_method`（v16）。更早的 CSV 整列不存在也能过校验，读作空。

> 「可为空」和「可缺列」不是一回事：`cross_check_rel_err` 必须**有这一列**但值可以空；`launch_signature` 连列都可以没有。这三组都由 `TestTheSchemaDocsCountsAreTheSchemas` 对着代码核——可为空那组是**拿校验器真跑出来的**（逐列置空，看 `validate_row` 收不收），不是另记一份名单。

| 列名 | 类型 | 来源 |
|---|---|---|
| `schema_version` | str | 例 `"5"`；本文件版本号，**整数字符串**，递增（详见规则 5）|
| `contributor` | str | 测量人 GitHub handle 或 org，例 `@anthropic` |
| `date` | str | ISO 8601 日期，例 `2026-05-16`；不带时间 |
| `sm` | str | 测量所用卡的 compute capability 标签，例 `sm_89`（Ada）/ `sm_90`（Hopper）/ `sm_100`（Blackwell）；由 `torch.cuda.get_device_capability` 得出。kernel 的可用性与性能是 SM-specific 的，故每行记录其实测架构 |
| `cross_check` | str | 该行延迟是否经独立 fp32 参考实现校验，未校验时说明原因。取值见下 |
| `cross_check_rel_err` | float 或空 | 该次比较的 `max|Δ| / mean|ref|`。空 = 没做过比较 |
| `cross_check_cosine` | float 或空 | 该次比较的余弦相似度，∈[-1,1]。空 = 没做过比较 |
| `xcheck_typical` | float 或空 | **判定依据之一**：`error = |Δ| / (|ref| + mean|ref|)` 的 RMS。空 = 没做过比较 |
| `xcheck_worst` | float 或空 | **判定依据之一**：同一 `error` 的最大值。空 = 没做过比较 |
| `kernel_symbol` | str 或空 | `<digest>:<name>` —— device 实际启动的 CUDA kernel 序列摘要，加上占用 device 时间最多的那个 kernel 名。空 = 未观测（详见下节）|
| `run_id` | str 或空 | 产生该行的那批 sweep 的 id（`hbench run` 每次调用生成一个，`run-all` 整批共享一个）。污染按批发现、按批撤销：一条撤销记录杀一个 run_id，历史不重写。空 = 该列存在之前测量的行 |
| `box` | str 或空 | 测量所用 box profile 名（`kernel_bench/envs/boxes/<name>.toml`）。箱况（clocks / power cap / driver）作出处记录、**不进 key**——共享包络的语义是"此 SKU 能做到的最好"，min over boxes 恰好是它。空 = 未解析到 profile |
| `measured_load_factor` | float 或空 | 测量所携带的真实路由分布的 max/mean 专家负载，≥ 1。声明先行：无 routing-trace 的行为空，不造默认值——一个编出来的 1.0 会被读成"实测均衡" |
| `input_seed` | str 或空 | 该行输入张量的抽取种子（`kernel_bench.invocation.Invocation.input_seed`）：十进制无符号 64 位整数，**以字符串存**，避免读者把它读成有符号或浮点。它由测量 key 的四个字段派生，所以可以从行上重算——"这一行是不是在今天的 harness 建的数据上测的"因此是一次查询而不是一次考古。**provenance 而非 identity**：key 已经说明这是哪个 cell，种子是它的函数。空 = 该列存在之前测量的行（那时的数据是全局 RNG 当时恰好持有的状态）|
| `measured_dtype_env_json` | JSON str 或空 | **kernel 实际计算所用的精度**，当它和 cell 要求的不同时——和 `dtype_env_json` 同形的 dtype_env，但**只列出不同的 slot**（因此不受规则 3 的「必须声明 activation/output」约束）。空 = kernel 按 cell 的精度计算（常态）。典型来源是**装载时反量化**：vLLM / sglang 在 `process_weights_after_loading` 把 absorbed MLA 投影（以及 DeepSeek-V4.1 的 grouped output LoRA）反量化成 bf16，再跑 `torch.bmm`——那是一个 fp8 checkpoint 的真实部署、服务器真付的代价，但算术是 bf16，延迟不是 fp8 kernel 的延迟。identity 仍然属于 **cell**（`dtype_env_json`）：cell 是 checkpoint 出厂的精度、也是估算问的那个精度，一个 cell 的两种配置应当落在同一个 key 上才能比较；本列说明某一行是哪一种配置。由 provider 的 `computes_at` 声明 |
| `launch_signature` | str 或空 | **被计时的那次宿主侧调用**，由它收到的实参渲染：`module.fn(<dtype>[<shape>], …, kw=…)`，张量非连续时带 `:strided`。空 = 该 provider 仍返回裸闭包。行上已经有 cell 的形状（`shape_json`）和设备侧真正跑的 kernel 序列（`kernel_symbol`），但都看不到**宿主把什么交给了库**——而那正是测量悄悄偏离 cell 的地方：`moe_permute` 的 cell 描述的是「读 token 序、写扩展后的专家序缓冲」的 dispatch，而被计时的调用是 `x[idx]`、两个操作数等长，流量比行的分子少 1.5 倍，于是那些行读出来是 HBM 峰值的 110%。**provenance 而非 identity**：同一个 cell 经两个库各自的调用约定仍是同一个 cell。由 provider 返回 `Launch(fn, *args, **kwargs)` 产生 |
| `outcome` | str 或空 | **这一行记录的是什么结果**，词表闭合（`OUTCOMES`，拼错即 reject）。空或 `"measured"` = 一次测量（该列之前写的每一行都是，之后绝大多数也是）。其余值是代替测量发生的事：`"oom"`（这张卡放不下这个 shape，对这个实现而言）、`"mismatch"`（跑出来了，但 cross-check 或 launch 门说它算的不是这个 cell）、`"error"`（provider 的 build / launch 抛异常，包括运行时报 unsupported——那说明它的声明漏了这一格）、`"crash"`（这个实现在跑时进程死掉，或 CUDA context 被打坏）、`"timeout"`（看门狗看到无进展把它杀掉，或单次 launch 超过计时预算）。**每一种（测量也一样）说的都是在这个实现上发生了什么**：`framework` / `framework_version` / `kernel_impl` 都是 identity，库升了版本就是另一个实现，会重试。行上看不见的是「同一库版本下 provider 自己的 bench 代码改了」——这对测量和失败完全一样，所以只有一条处理：修 provider build / 输入转换的那次合并里，按 `kernel_impl` 走既有作废通道清掉它的旧行（测量、oom、失败一起，见 `kernel_bench/ARCHITECTURE.md`）。**非 `"measured"` 的行一律不带测量列**（9 个 measurement 列必须为空，校验器只在这里允许为空；反之带了延迟就 reject），`cross_check` 记 `unknown`（它描述的是一次测量怎么被核过，而这里没有测量；mismatch 的判决就是 outcome 本身，依据是下面几列数字）。比较的四个数（`cross_check_rel_err` / `cross_check_cosine` / `xcheck_typical` / `xcheck_worst`）在非测量行上**只有 `mismatch` 可以带**（输出形状对不上时没有有限值，就不带，原因写在 `outcome_detail`）。被 harness 的覆盖判断读到（这个 cell × 实现在这个版本上已经有行 = 试过了，见 `kernel_bench/ARCHITECTURE.md`），**不**被估算读到（`op_calibration.registry_from_lookup` 跳过非 measured 的行，否则会去 `float("")`）|
| `outcome_detail` | str 或空 | **为什么这一行不是测量**，一行人读的话：异常（`RuntimeError: …`）、cross-check 的判决、看门狗的理由。非测量行**必须有值**，唯一例外是 `"oom"`（v15 之前的三个版本里写过的 OOM 行没有这一列，一行写完的数据无从追问当时卡上还剩多少显存；新的 OOM 行 driver 都会写），测量行**必须为空**（测量没有什么要解释的）。单行，最多 `OUTCOME_DETAIL_MAX`（240）字符。单设一列，是因为别的列都承载不了而不说谎：`launch_signature` 是被计时的调用、`kernel_symbol` 是 device 跑的 kernel、`cross_check` 是关于核验的闭合词表、`run_id` / `box` 是别人要 join 的出处。自由文本、从不被解析：类别看 `outcome`，这一列给读行的人 |
| `timing_method` | str 或空 | **哪只表计的这一行**，词表闭合（`TIMING_METHODS`）：`l2_drain`（GPU 默认：计时前只读扫一遍 2×L2 把 L2 扫干净，窗口内 launch 之后再扫一遍把它自己的脏行写回，扫描自身耗时单独测、逐样本减掉）/ `warm_l2`（`flush_l2=False`）/ `wall_clock`（无 GPU）。由计时函数自己随延迟一起给出。测量行为空 = **在这只表记录自己之前计的**，也就是旧的 memset 清 L2：launch 碰到的每一行都替清除缓冲区付一次写回，自己的写反而不付，小 kernel 的行可能偏差最多一个 L2 的写回量（L2 ÷ 显存带宽：H100 约 17 µs，RTX 4090 约 82 µs），方向不定。这些旧行按用户决定保留、不重测、不回填；Kernel Bench 页面在空值行上标注提示，不过滤。非测量行必须为空（没有计时）。provenance，不是 identity |
| `moe_block_member` | `"1"` 或空 | 该行是否属于一个 MoE 专家块——由声明的 dispatch zone 判定（zone 内节点，或它的 permute / combine 边界），plan 打在行上，block 聚合据此读、不再匹配 expert/moe 名。**可空以向后兼容**：schema-7 CSV 无此列，读作空，聚合暂无 MoE 块直到重新 emit |
| `kmd_version` | str 或空 | **这台机器的 kernel-mode 驱动版本**（`nvidia-smi` 打印的那串，如 `575.57.08` / `595.84`）。属于**机器**：同一台机上每个 env 看到的都一样。取自 `/proc/driver/nvidia/version`——内核模块自报，`$LD_LIBRARY_PATH` 动不了它。空 = 未观测到（v14 之前的行，或没有 NVIDIA 内核模块）|
| `umd_version` | str 或空 | **测量进程实际加载的 user-mode 驱动（`libcuda.so`）版本**，同一种版本形式。属于**进程**：取决于 loader 把 `libcuda.so.1` 解析到哪个文件，而那由 `$LD_LIBRARY_PATH` 决定。取自 `/proc/self/maps` 里被映射的那个路径——所以它是**真正在生效**的驱动，而不是某个 box profile 想提供的那个（overlay 没生效时这里如实报系统驱动）。空 = 未观测到 |

这两列只有**作为一对**才有意义，判定只有一处：`benchmarks.is_driver_substituted`。它只回答一个问题：**跑这一行的 user-mode 驱动是不是这台机器自己的那个。**

- 两者都已知且**相同** → `False`。常态：进程加载了自己内核模块配套的驱动。
- 两者都已知且**不同** → `True`。有别的 user-mode 驱动被插到了系统驱动前面。

**这一对不说、也不能说的两件事：**

1. **方向。** 插在前面的更新 → 那是 NVIDIA 的 forward compatibility；更旧的（陈旧 overlay）→ 同样是一次替换，但不是任何意义上的 "forward"。两者在这一对上完全一样，所以 `is_driver_substituted` 对两者都答 `True`，而**不许**有人把它读成 "forward"。
2. **是否受支持。** 那是**具体配对**的属性，不是「不相等」的推论。具体事实只有这一条：NVIDIA 为 `cuda-compat-13-x` 系列声明的 base install 底线是 **r580**，而 `kernel_bench/envs/boxes/h100-sxm-cu128.toml` 那台机的内核驱动是 **r575**，所以**那个**配置在支持矩阵之外（该 profile 里写明了，并附了交叉校验证据）。换一台机器的替换完全可能落在矩阵之内。行上既没有支持结论、也没有 `box`（`op_calibration.ROW_COLUMNS` 刻意不带），所以读这一对的人没有立场给出任何支持结论。
- 任一侧未知 → `None`，**不是 `False`**。缺驱动不是「原生」的证据；v14 之前的每一行都在这一档，给它们填一个推定值等于编造这对列本来要核的那件事。

**两列，不是三列。**「是否用了 compat 驱动」不另设一列，它就是 `umd_version != kmd_version`；也不在代码里列「哪些库挂 overlay」的名单——box profile 的 `[library_env]` 已经声明过一遍，消费方再列一份就是会漂的第二张表（CLAUDE.md §1.4）。

**provenance 而非 identity。** 旁边的 `box` 列已经把这一类定过性：箱况（clocks / power cap / **driver**）作出处记录、不进 key。进了 key 还会把语料库**永久劈开**而不是互相取代：v14 之前的行这两列为空、空即未知，于是「未知驱动」的 key 再也不会和「已知驱动」的 key 相遇。改为挂在 measured badge 上：`driver` 是真正执行了 kernel 的 user-mode 驱动，`driver_substituted_for` **仅在**两者不等时出现、值是它**不是**的那个（本机自己的 kernel-mode 驱动）。badge 不写方向、也不写支持结论。

`sm` 是 **provenance 而非 identity**：`gpu` 已经是 identity 列并锚定物理语料库的匹配键，`sm` 是与之一一对应的架构标注，新增它不改变任何 calibration 匹配行为。

`cross_check` 同样是 provenance：同一个 cell 在有参考和无参考两种条件下测出的是**一次**测量而非两次，故不进 identity。取值必须是下列之一：

| 值 | 含义 | 对应的修法 |
|---|---|---|
| `passed` | 参考实现可用，kernel 输出与之相符 | — |
| `no_reference` | 没有可用的 oracle：该 signature 未注册参考实现，或参考实现构造时 OOM | 补 `refs` 条目，或让参考实现省内存 |
| `not_comparable` | provider 未提供 `to_canonical`，输出无法归一化比较 | 给该 provider 补 `to_canonical` |
| `disabled` | 运行时传了 `--no-cross-check` | 操作者自己的选择 |
| `unknown` | 没有经过核验记录的行：schema v2 遗留行（早于本列存在），**或 `outcome` 不是 `measured` 的行**——那一行没有测量可核，结论看 `outcome`（mismatch 的判决就是它，依据在比较列） | v2 遗留行：重测；非测量行：按 `outcome` 处理，本列不需要改 |

**校验失败的行不会作为测量入库** —— driver 把它记成 `outcome="mismatch"` 的行（不带延迟，带判决依据的数字）。所以本列在测量行上区分的不是"对/错"，而是"查过/没查过"。这个区分无法从别处推断：fp32 oracle 的显存约为 kernel 的 2×、注意力还是 O(s_ctx²)，所以最大的那些 cell 上它会 OOM，kernel 照测但没有任何 oracle 把关；在本列出现之前，这类行和已验证的行在 CSV 里完全无法分辨。

v2 语料迁移到 v3 一律填 `unknown` 而非 `passed`（`tools/migrate_schema.py`）：v2 行本身不携带这个记录，而最可能未经校验的恰恰是那些最大的 attention / GEMM cell，把它们标成 `passed` 是一个关于从未发生过的检查的断言。

### 跑的是不是声明的那个 kernel（v5）

前三层保证证明声明格式合法、身份唯一、数字正确。**没有一层能区分"声明 A 跑了 A"和"声明 A 跑了 B"** —— 当 B 算的是同一件事时：FlashKDA 在 SM90+ 会静默接管 `chunk_kda`（除非 `FLA_FLASH_KDA=0`），vLLM 的 CustomOp 默认走 native torch 路径（除非 `custom_ops=all`）。两者都在错误的 kernel 名下产出正确的数字，而这正是数值校验看不见的那一种失败。

**先测量，再设计。** 原本的设计是"和声明的符号比对"，这不可行 —— 同一个 `kernel_impl` 在三个形状下是三个不同的 kernel 家族：

```
64x256x256       void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s16…>
4096x4096x14336  ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
512x2048x2048    ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn
```

cuBLAS 按形状选，连 `cutlass::`/`ampere_` 这个出处 token 都会变。所以本列**只记录，不声明期望值**，它支持的是两个不需要期望值的比较：

- 同一 signature 的两个 kernel 在**同一个 cell** 上启动了完全相同的序列 → 它们是同一个 kernel 顶着两个名字，一次测量记了两遍（`hbench smoke` 报告）；
- 同一个 `(kernel_impl, cell, gpu, version)` 在两次 run 之间 `kernel_symbol` 变了 → 发生了替换，比对行即可见。

torch 拼出来的参考实现启动的是一整个程序（`torch_rope` 有 18 个 kernel），所以单位是**序列**：digest 覆盖整个启动序列（任何一处替换都会让它变），名字取占用 device 时间最多的那个（这样人读一行就知道是谁的 kernel）。

观测在**计时区间之外**做一次：profiler 会把流串行化，放进计时区间会抬高它碰到的每一个延迟。profiler 挂不上的机器照常测量，本列留空 —— 空表示"没观测"，而不是"什么都没跑"。

### 判决之外，还要记下判决的依据（v4）

`cross_check` 只说"过了没有"。两个数一起才说得清**过成什么样**：

- `cross_check_rel_err = max|Δ| / mean|ref|`。它按构造在归约类算子（GEMM、attention）上偏大 —— 输出以零为中心，少数近零元素把比值抬起来 —— 所以**不可跨 signature 横比**。
- `cross_check_cosine` 与尺度无关，kernel 算对了它就贴近 1。
- **上面两列都不是判定依据。** 判定看 `xcheck_typical`（RMS）和 `xcheck_worst`（最大值），
  两者都必须落在该 dtype 的band 内（`driver._XCHECK_BANDS`，bf16 为 `Band(8e-3, 1e-1)`）；
  它们的分母含 per-element `|ref_i|`，`cross_check_rel_err` 不含，所以后者在动态范围宽的
  输出上按构造偏大。**`cross_check_rel_err` 大而 `cross_check='passed'` 是正常的**，
  不表示有分歧漏过了门 —— 全库 passed 行里它有 48% 超过 2e-2，且恰好集中在归约类
  （`per_head_linear` 89.5%、`flash_dsa` 84.3%、`grouped_gemm` 33.8%），而在无归约的
  `gather` / `add` / `moe_topk_softmax` 上是 0.0%。不要拿它和 band 比。

两个都留是因为**谁都不能替代谁**。fp8 的一条实测：`rel_err = 0.24` 单看像坏的，配上 `cosine = 0.9993` 一眼就是量化噪声；反过来，一个把结果整体放大 1.5× 的 kernel 余弦恒等于 1，只有 `rel_err` 抓得住它。

通过的行也记。事实上**只有通过的行会作为测量留下来**（失败的行记成 `mismatch`、不是测量），所以不在 pass 路径上算，这两个数就等于从不存在 —— v3 就是这样把它们算出来又丢掉的。这也是这条容差此前无法复核的原因：fp8 的带宽长期设在该 dtype 自身误差之下，表现出来只是"这个 provider 行数少"。

只有可能产生数字的 label 才允许带数字：`disabled` / `no_reference` / `not_comparable` 意味着比较从未发生，此时带数字的行要么标错了 label，要么把别的比较的数字带了过来，两种都会被读成"这个 kernel 查过"。校验器和 DB CHECK 都拦这一条。

v3 行迁移到 v4 一律留**空**，不回填 0：那次比较已经被丢掉，0 会被读成"完全一致"。

### 按批撤销与按箱归因（v7）

语料是事实层、append-only，"当前最优"是派生视图。这个模型需要两个前提列：

- **污染按批发生、事后发现**（16,045 行被并发污染的事故），所以删除的唯一正确形式是"一条撤销记录杀整批"——每行带 `run_id`，撤销不打断历史，视图自动排除。
- **共享包络的语义是"此 SKU 上此 kernel 能做到的最好"**，min over boxes 恰好是它；箱况只能作出处列供事后归因，进 key 会把同一物理格撕成多行。

`measured_load_factor` 是同一个"声明先行"的第三条：overlay 端要按模型×负载给经验分布，前提是语料能携带真实 routing trace 的统计。列先到位，producer 后补；无 trace 的行为空。

**严格白名单**：CSV header 必须是这 43 列（顺序不限，校验器按列名读）。多列、改名一律 reject；少列也 reject，**例外是 `moe_block_member`（v8 起）、`input_seed`（v9 起）、`measured_dtype_env_json`（v10 起）、`launch_signature`（v11 起）、`outcome`（v12 起）、`xcheck_typical` / `xcheck_worst`（v13 起）、`kmd_version` / `umd_version`（v14 起）、`outcome_detail`（v15 起）和 `timing_method`（v16 起）**——对更早的 CSV 可缺（校验器 `OPTIONAL_COLUMNS` 豁免，缺则读作空）。错误信息点出 offending column。

## 规则 2 —— `shape_json` symbol 限制

`shape_json` 是 JSON 字典字符串，例：

```json
{"B": 1, "s_ctx": 2048, "n_heads": 32, "d_head": 128}
```

允许出现的 key：

- **注册符号** —— `backend/arch/dims.yaml` 中 `symbols:` 的每一个 key。就是 spec 能绑定的那套名字，直接读注册表，不再从 markdown 表格里正则抠
- **Phase-binding 额外字段** —— `s_query` / `s_kv` / `s_cache` / `n_verify` / `s_past` / `s_pairs`（`benchmarks.PHASE_BINDING_EXTRAS`，由 `phases.py` 注入）
- **Engine 供给** —— `B` / `s_ctx`（engine 入口注入；常出现于实测 shape）

非以上集合的 key —— reject，错误信息形如 `unknown shape symbol 'X'; not in arch/dims.yaml, phase binding (…), or engine-supplied (…)`。

value 必须为正整数（int），float / 字符串 / 0 / 负数全部 reject。

## 规则 3 —— `dtype_env_json` 5-slot 限制

`dtype_env_json` 是 JSON 字典字符串，例：

```json
{"weight": "fp8_e4m3", "activation": "fp8_e4m3", "output": "bf16", "metadata": "fp32", "weight_granularity": "per-tile:128x128"}
```

允许出现的 key（**5-slot taxonomy**，与 M2 `access_bytes` 实现共享）：

| Slot key | 含义 |
|---|---|
| `weight` | 该行自己读的权重的 dtype；不读权重的行（attention、激活、collective）省略 |
| `activation` | activation dtype（输入和中间张量）|
| `kv` | 该行自己读 / 写的 cache 张量的 dtype；不碰 cache 的行省略 |
| `output` | output dtype；通常等于 `activation`，promote-precision 的 op 例外 |
| `metadata` | 该行权重自身 quant scale 的 dtype（如 FP8 scale_inv 是 FP32）；权重不带 scale 时省略 |

key 不在以上 5 个 —— reject。每个 slot 只写该行自己的事实，**不适用就省略**，从不借模型级的值（不借最宽 cache、最常见的 scale dtype，不拿 activation 顶 weight，也不补 BF16 / FP32）：`activation` 和 `output` 每行必须给出；`weight` 只在行读权重时出现；`kv` 只在行读写 cache 时出现；`metadata` 是该行权重自身 scale 的 dtype，权重不带 scale 就省略。带 scale 的行同时写 `weight_granularity`（哪些权重元素共享一个 scale），此时 `metadata` 必须给出；有 `metadata` 就必须有 `weight`。缺了必需的 slot 就 reject。`workload_hash` 对出现的 slot（含 `weight_granularity` / `activation_granularity` / `output_format` / `kv_format`）取值，省略的 slot 不参与，结果确定。`output` 是走访给这一行主输出定的 dtype（`LeafAccess.output_dtype`）；写一个 cached tensor 作为主输出的行，`output` 是该 cache 格式的元素 dtype。这类行若 cache 带 scale / transform / 不量化的尾段，同时写 `output_format`：该格式在 `backend/arch/cache_formats/` 里的名字（如 `fp8_ue8m0_group64_bf16_rope_tail` 是 DeepSeek-V4 的 FlashMLA 记录，`mxfp4` 是 MXFP4 index key）——元素 dtype 说不出 scale 怎么组织，两种记录在同一 dtype 上是两个 launch。名字必须在注册表里，且该格式的元素 dtype 就是 `output` 的 token（`format_refusal`）；主输出不是 cache、或 cache 只存 dtype 本身（无 scale）的行省略它。`kv_format` 是同一条规则用在 `kv` 槽上：`kv` 所指的那个 cache tensor（行读的，或在主输出之外写的）的格式名，元素 dtype 必须是 `kv` 的 token。写的一侧（vLLM DeepSeek-V4 的 q-norm/rope/KV-insert 在 query 旁写 FlashMLA 记录）和读的一侧（FlashMLA sparse decode 读同一份记录）都带它：读写同一种记录才是同一个 kernel，per-tensor FP8 的 cache 也是 `kv=fp8_e4m3`。没有 `kv` 的行、cache 只存 dtype 本身的行省略它。注册表没有名字的格式（精度 profile 里手写的 body）不生成 key，该行按 declined 处理。

value 必须是 **dtype 词表**里的字符串：

```
bf16, fp16, fp32, fp8_e4m3, fp8_e5m2, int8, int4, fp4_e2m1, mxfp4, mxfp8, ue8m0, int32
```

非词表 value —— reject，错误信息点出 offending value 和 slot。

## 规则 4 —— 同 `node_spec_id` + 不同 `shape_json` = 不同行

CSV 行身份是 **12 个 identity 列的全等元组**：

```
(model, spec_hash, phase, parallel, framework, framework_version, gpu,
 op, node_spec_id, shape_json, dtype_env_json, kernel_impl)
```

任何一列不同 —— 是不同行，独立测量。校验器看到全 12 列重复的两行 —— reject（duplicate identity）。

特别说明：assignment-mode `spec.model`（不同 `dim_overrides` 的同一模板）可能产出结构相同但 shape 不同的 leaves —— 它们的 `node_spec_id` 一样，但 `shape_json` 不同，因此是合法的两行。把 `shape_json` 留在 identity tuple 里就是为了不让这种情况被静默合并。

近邻匹配 / 跨 framework_version 复用 —— **不是 schema 层做**，是 Selection-mode 的事。validator 严格按全等。

## 规则 5 —— `schema_version` 是 forward-incompatible

`schema_version` 是整数字符串。validator 的 `KNOWN_SCHEMA_VERSION` 常量定义当前能识别的最高版本。

- CSV `schema_version` ≤ `KNOWN_SCHEMA_VERSION` —— 接受
- CSV `schema_version` > `KNOWN_SCHEMA_VERSION` —— **reject**（forward-incompatible by design）

要扩字段 / 改字段类型 —— bump `KNOWN_SCHEMA_VERSION` + 更新本文件 + 升 validator。**不要**保留 alias 兼容旧 schema —— 没有别名机制。

v6 起，`grouped_gemm` 的 `shape_json` 必须包含 `n_tokens`。旧行测量的
是完整 padded 几何，因此迁移工具可精确恢复为
`n_tokens = n_experts × s_expert_rows`；validator 不接受缺失该轴的旧身份。
2026-10-25 起的口径：收到 token 的专家才补齐，没收到的专家不算
（`touched_experts`）。cell 的 `n_experts` 是这一步的行实际到达的专家数：
每专家平均 ≥ 1 行时就是本 rank 的全部专家；不到 1 行时是 `ceil(E × M)`
（DeepSeek-V3 B=1 decode：8 个专家各 1 行，不是 256 个）。所以仍然
`n_tokens = n_experts × s_expert_rows`（`routed_launch_rows`）。
（2026-10-24 前该轴叫 `s_query`；让位给 dispatch 边界的 token 侧后，
routed cell 的每专家 padded 行数改名 `s_expert_rows`，`moe_permute` /
`moe_combine` 的 cell 同时带上 token 侧 `B × s_query`——旧边界行按旧拼写
保留、与新 key 不相交，随重新出 plan 重测。同日两个边界签名也带上
`n_tokens`，和 `grouped_gemm` 一样是路由 buffer 的总行数；其间测的边界行
按 `n_experts × s_expert_rows` 补齐。）

v7 起，provenance 增加 `run_id` / `box` / `measured_load_factor` 三个可空列。v8 起增加 `moe_block_member`（可空，向后兼容 schema-7）。v9 起增加 `input_seed`（可空，向后兼容 schema-8）。v10 起增加 `measured_dtype_env_json`（可空，向后兼容 schema-9）——精度 fallback 的标记，见字段表。v11 起增加 `launch_signature`（可空，向后兼容 schema-10）——被计时的宿主侧调用及其实参。v12 起增加 `outcome`（可空，向后兼容 schema-11）——这一行是测量还是一个被记录的结果（OOM）。v13 起增加 `xcheck_typical` / `xcheck_worst`（可空，向后兼容 schema-12）——cross-check 判定真正依据的那两个数。v14 起增加 `kmd_version` / `umd_version`（可空，向后兼容 schema-13）——**哪个驱动跑出了这一行**：机器的 kernel-mode 驱动，和测量进程实际加载的 user-mode 驱动。v15 把 `outcome` 从两个事实（`measured` / `oom`）扩到四种失败的尝试（`mismatch` / `error` / `crash` / `timeout`），并增加 `outcome_detail`（可空，向后兼容 schema-14）——失败的那一行话。此前失败不留行，于是「没人试过」和「试过、坏了」读起来一样，sweep 每一轮都花钱重新发现。v16 起增加 `timing_method`（可空，向后兼容 schema-15）——哪只表计的这一行；之前的行不迁移，读作空。
旧行一律迁为空：这些都是测量当下的事实，一份已完成的 CSV 无从追问。

当前版本：`KNOWN_SCHEMA_VERSION = 16`。

> 这个数和上面列数一样由 `TestTheSchemaDocsCountsAreTheSchemas` 对着 `benchmarks.KNOWN_SCHEMA_VERSION` 核。它曾经停在 12 而代码已经是 14 —— 而这是**最危险**的一种漂移：按本文件实现的外部生产者会把带驱动列的行盖上 `schema_version=12`，校验器**照收**（只拒更高的版本，不拒更低的），于是行的自述版本和它实际的列集合永久不一致。

## 规则 6 —— Round-trip byte-identity

validator 必须满足：parse 一个 well-formed CSV，原样 emit 回去，**字节级 identical**。

含义：
- header 列顺序由 input 决定（validator 不重新排序）
- 数值 / 字符串原文照搬（不做 normalization，例如不把 `1.0` 改成 `1`，不把 `2026-5-16` 改成 `2026-05-16`）
- JSON 字段（`shape_json` / `dtype_env_json`）按 input 字符串照搬（如果 input 是合法 JSON 即可；validator 不重新序列化）
- 行尾不补 trailing newline / 不改 line ending

如果 round-trip 不 byte-identical —— 校验器有 bug，**修校验器**，不要 fall back 接受。

## 规则 7 —— 生成 / 校验 工具

| 命令 | 用途 |
|---|---|
| `uv run python -m backend.cli bench validate <csv>` | 校验单个 CSV；exit 0 干净，非 0 失败 |
| `uv run python -m backend.cli bench suggest-shapes <model>` | 跑 `pipeline.analyze(resolved_spec, parallel=ParallelConfig())`，把 `AnalysisResult.per_op_timing` 的每一行翻成一个 CSV row 模板，identity 列尽量填，measurement / provenance 列留空 |
| Kernel Bench 上传（`hbench upload`、`/api/bench/upload`） | 入库前跑同一 validate，拒绝 schema 错 |

`suggest-shapes` 不维护 `INDEX.json` 或任何中间 artifact —— 直接读 `AnalysisResult`。

## 规则 8 —— 哪些事不归 CSV schema 管

- **t_op 的计算 / 校准** —— `op_calibration.measured_time_for(...)` 是 M4 的事；本 schema 只 freeze 数据格式，不绑解读
- **Source badges / Selection modes** —— UI surface，M5 的事；本 schema 只暴露原始数据
- **anomaly 检测**（`measured < roofline_at_test × 0.95`）—— M4 calibration overlay 做；CSV 只记录数值
- **submit helper** —— `huggingarch bench submit` 是 plan task15，optional follow-on；CSV schema 本身不需要

CSV schema 是**纯数据契约**：contributors 知道怎么写一行能通过 CI 的 CSV，calibration 层 / UI 层后续自由消费。
