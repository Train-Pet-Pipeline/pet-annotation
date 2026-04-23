# pet-annotation 架构文档

## §1 仓库职责

pet-annotation 是 Train-Pet-Pipeline 的 **4 范式打标引擎**，负责将 pet-data 产出的原始采样帧转化为结构化标注，供 pet-train 消费。

### 在流水线中的位置

```
pet-schema → pet-data → [pet-annotation] → pet-train → pet-eval → pet-quantize → pet-ota
```

### 做什么

1. **拉取 pending targets**：从 pet-data 的 `frames` 表（只读 URI 模式）拉取 `annotation_status='pending'` 的帧，注册到本仓库的 `annotation_targets` 状态机表。
2. **4 范式并发打标**：通过 `AnnotationOrchestrator` 为每个 `(target_id, annotator_id)` 组合分别调用 LLM / classifier / rule / human paradigm。
3. **写 4 范式表**：将 `LLMAnnotation / ClassifierAnnotation / RuleAnnotation / HumanAnnotation` 分别写入对应表（schema 来自 pet-schema）。
4. **导出 SFT/DPO 格式**：通过 `export/sft_dpo.py` 把 done 状态的标注导出为 JSONL，供 pet-train SFT trainer 和 DPO trainer 消费。

### 不做什么

- **不 ingest 原始数据**（pet-data 负责）
- **不训练模型**（pet-train 负责）
- **不写 pet-data 数据库**（只读跨仓访问）
- **不跨模型 reconcile**（D4 决策：每个 annotator 独立存储，不做 majority-vote 等融合）

---

## §2 I/O 契约

### 上游依赖

| 依赖 | 类型 | 版本 | 说明 |
|------|------|------|------|
| `pet-schema` | β peer-dep | v3.1.0 | Annotation 子类定义（4 范式 discriminator）|
| `pet-infra` | β peer-dep | v2.6.0 | DATASETS registry、structured logging |
| pet-data `frames` 表 | SQLite read-only | — | `annotation_status='pending'` 帧列表 |

β peer-dep：均不在 `pyproject.toml` dependencies；`_register.py` Mode B fail-fast guard；CI 5 步装序。

跨仓读取方式：`sqlite3.connect("file:<path>?mode=ro", uri=True, timeout=10)`（D1 决策，防止意外写入 pet-data）。

### 下游消费

| 消费方 | 消费方式 | 数据内容 |
|--------|---------|---------|
| pet-train | DATASETS plugin（`pet_annotation.llm` 等 4 个注册模块）| 4 范式标注表 |
| pet-eval | DATASETS plugin（同上）| 评估用标注 |
| pet-train / pet-eval（直接） | CLI `export --format sft/dpo` 产出 JSONL 文件 | SFT 样本 / DPO pair |

### Sample 类型

- `LLMAnnotation`：VLM/LLM 打标，含 `raw_response`, `parsed_output`, `prompt_hash`
- `ClassifierAnnotation`：本地分类器，含 `predicted_class`, `class_probs`, `logits`
- `RuleAnnotation`：确定性规则，含 `rule_id`, `rule_output`
- `HumanAnnotation`：Label Studio 人工审核，含 `reviewer`, `decision`, `notes`

### annotation_targets 状态机

表 `annotation_targets`（migration 005）以 `(target_id, annotator_id)` 为复合主键：

```
pending → in_progress → done
                     ↘ failed
```

- `pending`：从 pet-data 注册，尚未 claim
- `in_progress`：已被 orchestrator claim（BEGIN IMMEDIATE 原子操作），正在处理
- `done`：标注成功写入 4 范式表之一
- `failed`：标注出错，`error_msg` 字段存储错误描述

---

## §3 架构总览

### 目录树

```
src/pet_annotation/
├── teacher/
│   ├── orchestrator.py    ← 主 dispatch loop (async 4-paradigm sequential)
│   ├── provider.py        ← BaseProvider ABC
│   ├── providers/
│   │   ├── openai_compat.py  ← OpenAI-compatible provider (DashScope, OpenAI)
│   │   ├── vllm.py           ← vLLM provider
│   │   └── doubao.py         ← Doubao Vision provider
│   ├── cost_tracker.py    ← 日 token 预算追踪
│   └── rate_tracker.py    ← 请求速率限制
├── classifiers/
│   └── base.py            ← BaseClassifierAnnotator ABC + NoopClassifier
├── rules/
│   └── base.py            ← BaseRuleAnnotator ABC + BrightnessRule
├── human_review/
│   ├── ls_auth.py         ← Label Studio session/token auth
│   ├── ls_client.py       ← LS REST API wrapper (tenacity retry)
│   └── templates/         ← LS annotation template 配置
├── store.py               ← AnnotationStore (4 范式表 + annotation_targets 状态机)
├── adapter.py             ← annotator_type → store insert 方法路由
├── config.py              ← 4 范式 Pydantic 配置模型 + load_config()
├── cli.py                 ← CLI: annotate / export / stats
├── export/
│   ├── sft_dpo.py         ← JSONL 导出 (SFT + DPO) for pet-train
│   └── to_audio_labels.py ← 音频标签导出
├── dpo/
│   ├── import_app_feedback.py  ← app 反馈导入 DPO pair
│   └── validate_pairs.py       ← DPO pair schema 校验
├── datasets/
│   ├── llm_annotations.py       ← DATASETS plugin: pet_annotation.llm
│   ├── classifier_annotations.py← DATASETS plugin: pet_annotation.classifier
│   ├── rule_annotations.py      ← DATASETS plugin: pet_annotation.rule
│   └── human_annotations.py     ← DATASETS plugin: pet_annotation.human
├── quality/
│   ├── llm_judge.py       ← LLM-as-judge 质检
│   └── sampling.py        ← 随机抽样质检
└── _register.py           ← β peer-dep fail-fast guard + plugin 注册入口
migrations/
├── 001_create_annotation_tables.sql
├── 002_add_modality.sql
├── 003_create_audio_annotations.sql
├── 004_four_paradigm_tables.sql
└── 005_add_annotation_targets.sql   ← Phase 4 新增
params.yaml                ← 全部数值配置（batch_size/max_concurrent/threshold 等）
```

### 核心数据流

```
pet-data frames (annotation_status='pending')
    │
    ▼ ingest_pending_from_petdata() [read-only URI]
annotation_targets (state='pending')
    │
    ▼ claim_pending_targets() [BEGIN IMMEDIATE]
annotation_targets (state='in_progress')
    │
    ├─── _run_llm_paradigm()     → insert_llm()       → llm_annotations
    ├─── _run_classifier_paradigm() → insert_classifier() → classifier_annotations
    ├─── _run_rule_paradigm()    → insert_rule()      → rule_annotations
    └─── _run_human_paradigm()   → insert_human()     → human_annotations
         (submit to LS + pull completed, two-phase)
    │
    ▼ mark_target_done() / mark_target_failed()
annotation_targets (state='done'|'failed')
    │
    ▼ to_sft_samples() / to_dpo_pairs()
JSONL (SFT / DPO) → pet-train
```

---

## §4 核心模块详解

### 4.1 `teacher/orchestrator.py` — async 4-paradigm dispatch

**Why**：Phase 2 BREAKING 重构后各范式 dispatch 是独立 stub，Phase 4 完成 wire。`AnnotationOrchestrator.run()` 统一 ingest → claim → annotate → insert → mark_done 模式，跨 4 个范式结构对称，减少认知负担。

**Tradeoff**：4 个范式在 `run()` 内**顺序**执行（llm → classifier → rule → human），而非 `asyncio.gather` 并发——简单、可预测、信号优雅关闭（`self._shutdown` flag 在范式边界处检测）。单范式内部的 target 批次通过 `asyncio.gather(*tasks)` 并发处理，受 `asyncio.Semaphore(max_concurrent)` 控制吞吐。

**Pitfall**：`asyncio.gather` 任务共享同一个 `sqlite3.Connection`（`store._conn`）。Python 的 sqlite3.Connection **不是协程安全的**：若 coroutine A 的 insert 和 coroutine B 的 commit 交错，会破坏事务边界，可能导致数据丢失。Phase 4 review 将此列为 CRITICAL 并修复——在 `_process_one` / `_process_one_classifier` / `_process_one_rule` 的写操作处统一用 `async with self._write_lock`（`asyncio.Lock`）序列化写路径。

关键代码路径（`orchestrator.py:130`）：

```python
# Lock serializes sqlite3 writes across asyncio tasks sharing one connection.
self._write_lock = asyncio.Lock()
```

### 4.2 `store.py` — AnnotationStore + annotation_targets 状态机

**Why**：`annotation_targets` 表的 `(target_id, annotator_id)` 复合主键，配合 4-state CHECK 约束，让 1..N 模型天然独立（D3 决策）——annotator A 和 annotator B 对同一 target 各自有独立的状态行，互不干扰（D4 决策：不跨模型 reconcile）。`ingest_pending_from_petdata` 用 `mode=ro` URI 跨仓读 pet-data，INSERT OR IGNORE 保证幂等（D1 决策）。

**Tradeoff**：pet-annotation 维护自己独立的 SQLite，**不写** pet-data（D2 决策）——pet-data 的 `annotation_status` 字段保留给 manual QA 用；pet-annotation 状态与 pet-data 完全解耦，方便各自独立演进。代价是两个数据库之间无外键约束，一致性靠 CI 和运维检查。

**Pitfall**：`ingest_pending_from_petdata` 的 pet-data 路径来自 `config.annotation.pet_data_db_path`（params.yaml 字段）。若配置错误或文件不存在，会抛出 `sqlite3.OperationalError`（连接时立即失败，不会静默）。`claim_pending_targets` 使用 `BEGIN IMMEDIATE` 防止并发 double-claim，但 sqlite3 WAL 模式下同一进程单连接不存在并发写——这是为未来可能的多进程部署预留的防护。

### 4.3 4 范式插件接口

**Why**：4 个范式本质不同（LLM 异步网络调用 / classifier 本地同步推理 / rule 纯函数 / human 异步双阶段 LS 交互），但统一接口：provider/plugin 输出 → pet-schema Annotation 子类 → 写对应表。接口统一让 orchestrator 的 claim/mark_done 逻辑复用，也让测试 mock 更简单。

**Tradeoff**：classifier 和 rule 是同步调用，通过 `asyncio.to_thread()` 包装成协程——统一了 async 调度框架，代价是每次调用有 ThreadPool 开销（对于 tiny rule 函数有浪费）。这个代价可接受，因为吞吐瓶颈在 LLM API 网络延迟，不在 rule 执行时间。

**Pitfall**：
- `BaseClassifierAnnotator.annotate()` 返回 `(predicted_class, class_probs, logits)`；logits 可以为 `None`（并非所有 classifier 都暴露 logit）。store 写入时 `_dumps(ann.logits) if ann.logits is not None else None`，读取时用 `_loads`——NULL 和空 list 含义不同，不能混淆。
- `BaseRuleAnnotator.apply()` 约定：规则未触发时返回 `{}`（空 dict），**不是** `None`。store 和 export 均用 `_dumps(ann.rule_output)` 序列化，空 dict `{}` 合法。
- `HumanAnnotator` **不阻塞**等人工标注完成：`_run_human_paradigm` 在一次 `run()` 中完成"提交 batch 给 LS"（Phase A）和"拉取 LS 已完成标注"（Phase B）两个阶段。已提交但未完成的 target 保持 `in_progress` 状态，等下次 `run()` 的 Phase B 再拉取。

### 4.4 `human_review/ls_client.py` — Label Studio REST wrapper

**Why**：LS 是唯一的外部有状态服务（其他所有 plugin 均本地运行）。需要处理 session auth、HTTP retry、task ID 映射三个独立关注点，单独封装在 LSClient 中。

**Tradeoff**：tenacity retry 策略只对 `ConnectionError / Timeout / 429 / 5xx` 重试，**不对 4xx** 重试（`_is_retriable_http_error` in `ls_client.py:31`）——避免 auth 失败（401）或 bad body（400）的无效重试浪费；`stop_after_attempt(3)` + `wait_exponential(1, min=1, max=10)`，最坏等待 ~1+2+4=7 秒。`fetch_completed_annotations` 固定 `page_size=1000`——MVP 实现，超过 1000 completed tasks 时会漏数据（§9 已登记）。

**Pitfall**：LS task ID（整数，LS 自动分配）≠ target_id（本仓库 frame_id 字符串）。映射关系在 submit 时通过 `"meta": {"target_id": tid}` 埋入 task，在 pull 时通过 `task["meta"]["target_id"]` 取回（`orchestrator.py:521`）。若 task 的 meta 字段缺失 target_id，该 task 被 warn-log 并跳过，不影响其他 task 处理。

### 4.5 `export/sft_dpo.py` — JSONL 导出

**Why**：pet-train SFT trainer 消费按 annotator_type 分列的 SFT 样本；pet-train DPO trainer 消费 chosen/rejected pair。JSONL 是 Python 生态最通用的行式格式，支持流式读取，单文件即可。

**Tradeoff**：export 直接访问 `store._conn` 执行 SQL JOIN，而不是走 AnnotationStore 的 public fetch 方法——快、简洁，JOIN 在 SQLite 内完成。代价是耦合 store 内部连接实现；若 store 未来改为连接池或读写分离，export 需要同步修改（§9 followup）。当前 export 是低频调用（不是热路径），这个 tradeoff 可接受。

**Pitfall**：4 个范式的 schema 不完全对齐——LLM 有 `raw_response` 字段，classifier 没有；rule 有 `rule_id`，human 有 `reviewer/decision/notes`。`to_sft_samples` 按 `annotator_type` 分支，各自把标注内容序列化为 `output` 字段的 JSON 字符串。`to_dpo_pairs` 对 LLM 范式做真实 pair（同 target 多 annotator，按 `confidence_overall` 排序 chosen/rejected）；对其他 3 个范式，DPO pair 的 chosen == rejected（自对，供下游过滤）。

---

## §5 扩展点

### 5.1 添加新 LLM annotator 模型

修改 `params.yaml` 的 `llm.annotators` list，追加一项（不改代码）：

```yaml
llm:
  annotators:
    - id: "my-new-model"
      provider: "openai_compat"      # or "vllm"
      base_url: "https://api.example.com/v1"
      model_name: "my-model-name"
      temperature: 0.1
      max_tokens: 2048
      api_key: ""                    # or set to env var value
```

### 5.2 添加新 LLM provider（非 OpenAI-compat）

新建 `src/pet_annotation/teacher/providers/<name>.py`，继承 `OpenAICompatProvider` 或直接实现 `BaseProvider.annotate()` async 接口（返回 `AnnotationResult(raw_response, prompt_tokens, completion_tokens)`）。在 `_build_provider()` (`orchestrator.py:61`) 中按 `llm_cfg.provider` 值分发到新 provider 类。

### 5.3 添加新 classifier plugin

1. 新建 `src/pet_annotation/classifiers/<name>.py`，继承 `BaseClassifierAnnotator`，声明 `plugin_name: ClassVar[str]`，实现 `annotate()` 方法。
2. 在 orchestrator 的 `_classifier_plugins` dict 中注入（生产代码在构建 orchestrator 后通过 `orch._classifier_plugins[plugin_name] = MyClassifier()` 注入，或实现 `_build_classifier_plugins()` 工厂方法）。
3. 在 `params.yaml` `classifier.annotators` 中配置对应的 `plugin` 字段。

### 5.4 添加新 rule

1. 新建 `src/pet_annotation/rules/<name>.py`，继承 `BaseRuleAnnotator`，声明 `rule_id: ClassVar[str]`，实现 `apply()` 方法（纯函数，返回 JSON-serializable dict）。
2. 在 orchestrator `_rule_plugins` dict 中注入。
3. 在 `params.yaml` `rule.annotators` 中配置 `plugin` 和 `rule_id` 字段。

### 5.5 添加新 human paradigm backend（非 LS）

实现与 `LSClient` 相同接口的类（`submit_tasks(tasks) -> list[int]` 和 `fetch_completed_annotations() -> list[dict]`），在 `_run_human_paradigm` 的 `ls_client = LSClient(...)` 处按 `human_cfg` 中新字段（如 `backend_type`）分发到新 backend。

### 5.6 添加第 5 种 paradigm

1. 在 **pet-schema** 添加新 `Annotation` 子类（需 pet-schema major/minor bump）。
2. 在本仓库新增 migration SQL（如 `006_create_fifth_paradigm.sql`），创建对应的 `fifth_paradigm_annotations` 表。
3. 在 `store.py` 添加 `insert_fifth_paradigm()` 和 `fetch_fifth_paradigm_by_target()` 方法。
4. 在 `orchestrator.py` 新增 `_run_fifth_paradigm()` 方法，并在 `run()` 中追加调用。
5. 在 `adapter.py` 的 `_ROUTES` dict 添加路由条目。
6. 新增 DATASETS plugin 文件 `datasets/fifth_paradigm_annotations.py`。
7. 在 `_register.py` `register_all()` 中 import 新 plugin 模块。

---

## §6 依赖管理

### β peer-dep（pet-schema + pet-infra）

pet-schema v3.1.0 和 pet-infra v2.6.0 均为 **β peer-dep**，**不出现**在 `pyproject.toml` 的 `dependencies` 中。安装顺序和 fail-fast guard 由 `_register.py` 的 Mode B 机制保证：

```python
# _register.py: Mode B delayed fail-fast
try:
    import pet_schema.version as _psv
except (ImportError, ModuleNotFoundError) as e:
    raise RuntimeError("pet-schema required ...") from e
```

CI 5 步装序（`.github/workflows/` 中定义）：
1. 安装 pet-infra（固定 tag v2.6.0）
2. 安装 pet-schema（固定 tag v3.1.0）
3. 安装 pet-annotation（`pip install -e .`）
4. 运行 `make lint`
5. 运行 `make test`

版本 pin 存储在 `src/pet_annotation/_version_pins.py`（`PET_SCHEMA_PIN = "v3.1.0"`, `PET_INFRA_PIN = "v2.6.0"`），测试 `tests/test_version.py` 验证 pin 值一致性。

### 跨仓读（pet-data）

pet-data SQLite 文件路径通过 `params.yaml` 的 `annotation.pet_data_db_path` 配置，CLI `--pet-data-db` flag 可在运行时覆盖。所有读取均通过：

```python
sqlite3.connect(f"file:{pet_data_db_path}?mode=ro", uri=True, timeout=10)
```

mode=ro 在 SQLite 层面强制只读（D1 决策）；若路径不存在或文件不是 sqlite，会立即抛 `sqlite3.OperationalError`（不静默失败）。

### 第三方依赖

| 包 | 用途 |
|----|------|
| `tenacity>=8.0` | LS HTTP retry（`ls_client.py`） |
| `requests>=2.0` | LS REST API HTTP 客户端 |
| `click>=8.0` | CLI（`cli.py`） |
| `pydantic>=2.0` | 配置模型（`config.py`）+ schema 验证 |
| `pyyaml>=6.0` | params.yaml 加载 |
| `aiohttp>=3.9` | provider async HTTP（teacher/providers）|

---

## §7 本地开发与测试

### 环境

使用共享 conda env `pet-pipeline`（**不建**仓库专属 env）：

```bash
conda activate pet-pipeline
```

### 安装顺序（β peer-dep 必须先装）

```bash
pip install "pet-infra @ git+https://github.com/Train-Pet-Pipeline/pet-infra@v2.6.0"
pip install "pet-schema @ git+https://github.com/Train-Pet-Pipeline/pet-schema@v3.1.0"
pip install -e ".[dev]"
```

### 常用命令

```bash
make setup    # 等价于上述 3 步安装
make test     # pytest tests/ (195 tests, ~30s)
make lint     # ruff check + ruff format --check
make clean    # 清理 .pyc / __pycache__ / *.db
```

### 测试架构

- **unit/**：每模块独立单元测试（store、config、classifiers、rules、adapter、export、ls_client 等）
- **integration/**：每范式端到端测试（llm/classifier/rule/human）
- **E2E 测试**：`tests/test_e2e_four_paradigm.py`，覆盖全 4 范式 + export JSONL，使用 `tmp_path` sqlite fixture，不依赖外部服务

总计 195 tests / 0 lint errors（Phase 4 final state）。

---

## §8 已知复杂点（复杂但必要）

### 8.1 asyncio.gather 共享 sqlite3 连接的竞态保护（self._write_lock）

**保留理由**：`AnnotationStore` 持有单个 `sqlite3.Connection`（`store._conn`）。asyncio 的并发 coroutine 在同一线程内交错执行，若 coroutine A 的 `execute(INSERT)` 和 coroutine B 的 `commit()` 交错，sqlite3 内部事务边界会错乱，可能静默丢数据或触发 `ProgrammingError`。`asyncio.Lock` 是正确的协程间互斥方案（不涉及线程，`threading.Lock` 无法保护协程交错）。

**删了会损失什么**：并发 batch 处理时数据写入随机丢失，事务边界错乱，极难复现的 heisenbug；对 1..N annotator 并发场景尤其危险。

**重新审视触发条件**：若未来改用 PostgreSQL（真连接池，连接天然隔离）或每个 coroutine 使用独立 sqlite3.Connection，可以去掉 write_lock。

### 8.2 4 paradigm 顺序执行（而非并发 paradigm）

**保留理由**：`run()` 内 llm → classifier → rule → human 顺序执行，配合 `self._shutdown` flag 在每个范式开始前检查，确保 SIGINT/SIGTERM 可以在范式边界干净退出，不会留下半完成的数据。实际吞吐由单范式内的 `asyncio.gather(batch)` + `semaphore(max_concurrent)` 控制，范式级并发收益微小（范式间无数据依赖）。

**删了会损失什么**：`asyncio.gather` 并发 4 个范式需要统一的 shutdown 协调机制（cancelled coroutine 中的 write_lock 持有情况更复杂），编码复杂度换来的吞吐提升不到 10%（LLM 范式是主要瓶颈，网络 IO 决定吞吐）。

**重新审视触发条件**：若范式数量增长到 > 10 个，或单个范式耗时极长（> hours），范式间并发才有实质价值。

### 8.3 export 直连 store._conn（而非公开 iter API）

**保留理由**：`export/sft_dpo.py` 中 `_iter_done_llm_rows()` 等函数直接访问 `store._conn.execute(SQL JOIN)`，而不是通过 AnnotationStore 的 `fetch_*_by_target()` 方法（后者是按 target_id 逐行读，无法高效做 JOIN + 状态过滤）。SQL JOIN 在 SQLite 内一次完成，性能好；export 是低频批量操作（每次训练前调一次），不是热路径。

**删了会损失什么**：若改为公开 `iter_done_annotations(annotator_type)` API，代码解耦更好，但需要 AnnotationStore 新增方法 + 接口定义，当前无其他消费方，过度设计。

**重新审视触发条件**：若 AnnotationStore 内部改架构（改用连接池、读写分离、或迁移到 PostgreSQL），export 模块需同步修改 —— 届时引入公开 iter API 是正确重构点。

---

## §9 Phase 5+ Followups

以下事项为 MVP 遗留，非 bug，触发条件明确时处理：

1. **LSClient 分页支持** — 触发：单 LS project 超过 1000 completed tasks；当前 `page_size=1000` 硬限，超出会漏数据。影响 human paradigm 数据完整性。

2. **LS incremental pull（updated_after 持久化）** — 触发：真实 LS 部署规模扩大（tasks > 几千）；当前每次 `fetch_completed_annotations()` 全量拉取，浪费网络且慢。修复：持久化上次 pull 时间戳，每次只拉 `updated_at__gt` 后的增量。

3. **Export module 走公开 iter API** — 触发：AnnotationStore 内部架构变动（连接池 / 读写分离 / 迁移 Postgres）；当前直连 `store._conn` 耦合内部实现。

4. **storage_uri 严格性验证** — 当前 `_fetch_storage_uri` 在 frame 不存在或列不存在时 None fallback（用 target_id 代替），不报错。触发：生产部署需要确保所有帧都有 storage_uri；可引入 CLI `--strict-storage` flag 让 None fallback 改为报错。

5. **Human paradigm 多 backend 支持** — 触发：除 Label Studio 外引入 Prolific 或自研标注 UI；当前 `_run_human_paradigm` 硬编码 LSClient。需要 backend factory 按 `human_cfg.backend_type` 分发。

6. **CLI `_call_provider` 参数 rename** — `orchestrator._call_provider` 的 `target_id` 参数实际传入的是 `image_path`（storage_uri resolved），命名有误导性；可 rename 为 `image_path` 提升可读性。

7. **`cli.py` 重复 json import cleanup** — `export_cmd` 内部 `import json as _json` 在 if/else 两个分支各 import 一次，cosmetic 问题，可提取到函数顶部。

8. **API token env var 缺失时 warn log** — `_run_human_paradigm` 中 `api_token = _os.environ.get(human_cfg.ls_api_token_env, "")` 在 token 缺失时静默返回空字符串，后续 auth 会失败（RuntimeError）。建议在 token 为空时提前 warn log，改善 UX。

9. **每 annotator timeout/max_retries 配置** — 当前 `LLMAnnotatorConfig` / `ClassifierAnnotatorConfig` 无 per-annotator timeout 字段，只有 params.yaml 级别的 model timeout。触发：不同 annotator 需要不同重试策略（如生产 LLM vs. 本地快速 classifier）。

10. **Provider session cleanup（aiohttp 未显式 close）** — `OpenAICompatProvider` / `VLLMProvider` 使用 aiohttp，短进程不影响，但长运行进程（如 daemon 模式）会有连接泄漏。触发：pet-annotation 作为长运行服务部署时，需要 `aiohttp.ClientSession.__aexit__` 显式关闭。
