# pet-annotation 设计文档

> **状态**：已评审通过  
> **日期**：2026-04-15  
> **作者**：Claude Code + 项目负责人  
> **上游依赖**：pet-schema v1.0.0, pet-data v1  
> **下游消费方**：pet-train

---

## 1. 概述

pet-annotation 消费 pet-data 的 `frames` 表中的帧图像，通过云端 VLM API 进行批量打标，产出 SFT（sharegpt 格式）和 DPO（chosen/rejected 格式）训练数据，供 pet-train 消费。

**核心增强需求（相对于 DEVELOPMENT_GUIDE 原有设计）**：

- 多账号 + 异步高速打标：同一模型配置多个 API key，速率感知调度，asyncio 并发
- 多模型并行打标：同一帧由多个模型分别打标，方便横向对比
- 请求级缓存：断点续跑时跳过已完成帧，避免重复 token 消耗

**架构选择**：统一调度器架构（AnnotationOrchestrator），单进程 asyncio + 线程池混合模式。选择理由见 §2。

---

## 2. 架构决策

### 2.1 为什么选统一调度器而非管道/队列

| 方案 | 优点 | 缺点 | 适用规模 |
|---|---|---|---|
| **统一调度器（选定）** | 调度逻辑集中，多模型对比容易实现 | Orchestrator 职责较重 | 万级帧 |
| DVC 管道阶段 | 与 pet-data 风格一致 | 多模型并发困难，跨进程调度复杂 | 千级帧 |
| 消息队列 + Worker | 天然横向扩展 | SQLite 并发写瓶颈，当前规模过度设计 | 百万级帧 |

### 2.2 多模型策略：主从模式

- **主模型**（`primary_model`）：走完整标注状态机，结果写入 `annotations` 表
- **旁路模型**：结果写入 `model_comparisons` 表，只存储不影响主流程
- 选择理由：pet-data 的 `frames.annotation_status` 只有一列，多轨道会破坏上下游契约

### 2.3 并发模型：asyncio + 线程池混合

- API 调用：asyncio + aiohttp，单线程内高并发
- SQLite 读写：`run_in_executor` 卸载到单 worker 线程池（避免 SQLite 写竞争）
- 选择理由：API 调用是 IO 密集型瓶颈，asyncio 最高效；SQLite 不支持原生 async

### 2.4 账号调度：速率感知

- 每个 API key 维护 60 秒滑动窗口，追踪 RPM/TPM 实际用量
- `acquire()` 选择余量比值最低的 key
- 所有 key 满载时 await 等待，不返回错误

### 2.5 跨仓库数据库访问模式

pet-annotation 需要操作 pet-data 的 SQLite 数据库（读 `frames` 表、更新 `annotation_status`、写 `annotations`/`model_comparisons` 表）。访问模式约定：

- **pet-annotation 的 `store.py` 创建独立的 `AnnotationStore` 类**，接收 pet-data 数据库文件路径，打开自己的 SQLite 连接
- **表的归属边界**：`frames` 表由 pet-data 的 `FrameStore` 创建和管理；`annotations`/`model_comparisons` 表由 pet-annotation 创建和管理
- **pet-annotation 对 `frames` 表只做 `SELECT` 和 `UPDATE annotation_status`**，不做 DDL 或其他列修改
- **建表方式**：`AnnotationStore.__init__` 中执行 `CREATE TABLE IF NOT EXISTS`（SQL 脚本在 `migrations/001_create_annotation_tables.sql`）。不使用 Alembic（pet-annotation 不拥有 `frames` 表的迁移权）
- **WAL 模式**：两个连接都使用 WAL 模式（SQLite 允许 WAL 下多连接并发读，写仍串行）。pet-data 的 `FrameStore` 已启用 WAL
- **并发保护**：pet-annotation 和 pet-data 不应同时对同一行做写操作。实际上 pet-data 的数据采集完成后才进入 annotation 阶段，时序上不冲突。如果出现 `SQLITE_BUSY`，tenacity 重试机制覆盖
- **数据库路径**：从 `params.yaml` 的 `database.path` 字段读取

### 2.6 日志规范

遵循开发指南要求，使用结构化 JSON 日志格式：
- 所有模块通过 `logging.getLogger(__name__)` 获取 logger
- 根 logger 配置 JSON formatter（在 `config.py` 中初始化）
- API 调用日志包含 `model_name`、`frame_id`、`latency_ms`、`tokens` 等结构化字段

---

## 3. 项目结构

```
pet-annotation/
├── src/
│   └── pet_annotation/
│       ├── __init__.py
│       ├── config.py                  # 加载 params.yaml，Pydantic Settings
│       ├── store.py                   # AnnotationStore（独立连接 pet-data 的 SQLite，见 §2.5）
│       ├── teacher/
│       │   ├── __init__.py
│       │   ├── provider.py            # Provider 抽象基类 + ProviderRegistry
│       │   ├── providers/
│       │   │   ├── __init__.py
│       │   │   ├── openai_compat.py   # OpenAI 兼容 API（Qwen/通义千问等）
│       │   │   ├── doubao.py          # 豆包（火山引擎 API）
│       │   │   └── vllm.py            # 自建 vLLM endpoint
│       │   ├── rate_tracker.py        # 速率感知调度（RPM/TPM 追踪）
│       │   ├── cost_tracker.py        # token 用量追踪 + 每日上限告警
│       │   ├── cache.py               # 请求级缓存（帧ID+模型+prompt_hash）
│       │   ├── batch_runner.py        # 异步批量推理入口
│       │   └── orchestrator.py        # 统一编排器
│       ├── quality/
│       │   ├── __init__.py
│       │   ├── auto_check.py          # schema 验证 + 代码层校验
│       │   ├── llm_judge.py           # LLM-as-Judge 一致性检查
│       │   └── sampling.py            # 置信度抽样 → needs_review
│       ├── human_review/
│       │   ├── __init__.py
│       │   ├── sft_config.xml         # Label Studio SFT 审核界面
│       │   ├── dpo_config.xml         # Label Studio DPO Pairwise 界面
│       │   ├── import_to_ls.py        # VLM 输出 → Label Studio task
│       │   └── export_from_ls.py      # 审核结果 → 数据库
│       ├── dpo/
│       │   ├── __init__.py
│       │   ├── generate_pairs.py      # 跨模型配对 + 教师-学生配对
│       │   ├── import_app_feedback.py # APP 用户纠错 → Label Studio
│       │   └── validate_pairs.py      # DPO 对五条合法性验证
│       └── export/
│           ├── __init__.py
│           ├── to_sharegpt.py         # 导出 SFT sharegpt JSONL
│           ├── to_dpo_pairs.py        # 导出 DPO JSONL
│           └── to_audio_labels.py     # 音频分类标签导出
├── tests/
│   ├── conftest.py
│   ├── test_provider.py
│   ├── test_rate_tracker.py
│   ├── test_cache.py
│   ├── test_orchestrator.py
│   ├── test_auto_check.py
│   ├── test_sampling.py
│   ├── test_generate_pairs.py
│   ├── test_validate_pairs.py
│   ├── test_export.py
│   ├── test_store.py
│   └── test_config.py
├── migrations/
│   └── 001_create_annotation_tables.sql  # annotations + model_comparisons 建表
├── params.yaml
├── dvc.yaml
├── pyproject.toml
├── Makefile
├── .env.example                       # 所有需要的环境变量模板
└── requirements.in                    # pip-compile → requirements.txt
```

---

## 4. 配置设计（params.yaml）

```yaml
database:
  path: "/data/pet-data/pet_data.db"    # pet-data 的 SQLite 数据库路径
  data_root: "/data/pet-data"            # 帧图像的根目录（frame_path 相对于此）

annotation:
  batch_size: 16                         # 每批拉取帧数
  max_concurrent: 50                     # asyncio 最大并发 API 请求数（新增，DEVELOPMENT_GUIDE 未定义）
  max_daily_tokens: 10_000_000
  review_sampling_rate: 0.15
  low_confidence_threshold: 0.70
  primary_model: "qwen2.5-vl-72b"       # 主模型名（新增）
  schema_version: "1.0"                  # 对应 pet-schema 版本（新增）

models:
  qwen2.5-vl-72b:
    provider: "openai_compat"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: "qwen2.5-vl-72b-instruct"
    accounts:
      - key_env: "QWEN_API_KEY_1"
        rpm: 60
        tpm: 100000
      - key_env: "QWEN_API_KEY_2"
        rpm: 60
        tpm: 100000
    timeout: 60
    max_retries: 3

  doubao-vision:
    provider: "doubao"
    base_url: "https://ark.cn-beijing.volces.com/api/v3"
    model_name: "doubao-vision-pro-32k"
    accounts:
      - key_env: "DOUBAO_API_KEY_1"
        rpm: 30
        tpm: 50000
    timeout: 60
    max_retries: 3

  local-vllm:
    provider: "vllm"
    base_url: "http://localhost:8000/v1"
    model_name: "Qwen/Qwen2.5-VL-72B-Instruct"
    accounts:
      - key_env: ""
        rpm: 999
        tpm: 999999
    timeout: 120
    max_retries: 2

dpo:
  min_pairs_per_release: 500

dvc:
  remote: "local"
  remote_path: "/data/dvc-cache"
```

**关键约束**：
- API key 只存环境变量名，明文 key 放 `.env`（已在 `.gitignore` 中）
- 每个 account 声明自己的 rpm/tpm 限额，RateTracker 据此调度
- `primary_model` 决定走状态机的模型，必须在 `models` 中存在

---

## 5. 数据库 Schema 扩展

在 pet-data 的 SQLite 数据库中新增两张表：

### 5.1 annotations 表（主模型结果）

```sql
CREATE TABLE annotations (
    annotation_id     TEXT PRIMARY KEY,
    frame_id          TEXT NOT NULL REFERENCES frames(frame_id),
    model_name        TEXT NOT NULL,
    prompt_hash       TEXT NOT NULL,
    raw_response      TEXT NOT NULL,
    parsed_output     TEXT,
    schema_valid      INTEGER NOT NULL,
    validation_errors TEXT,
    confidence_overall REAL,
    review_status     TEXT NOT NULL DEFAULT 'pending'
        CHECK(review_status IN ('pending','approved','needs_review','reviewed','rejected')),
    reviewer          TEXT,
    review_notes      TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    api_latency_ms    INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(frame_id, model_name, prompt_hash)
);

CREATE INDEX idx_annotations_frame  ON annotations(frame_id);
CREATE INDEX idx_annotations_model  ON annotations(model_name);
CREATE INDEX idx_annotations_review ON annotations(review_status);
CREATE INDEX idx_annotations_conf   ON annotations(confidence_overall);
```

### 5.2 model_comparisons 表（旁路模型结果）

```sql
CREATE TABLE model_comparisons (
    comparison_id     TEXT PRIMARY KEY,
    frame_id          TEXT NOT NULL REFERENCES frames(frame_id),
    model_name        TEXT NOT NULL,
    prompt_hash       TEXT NOT NULL,
    raw_response      TEXT NOT NULL,
    parsed_output     TEXT,
    schema_valid      INTEGER NOT NULL,
    validation_errors TEXT,
    confidence_overall REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    api_latency_ms    INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(frame_id, model_name, prompt_hash)
);

CREATE INDEX idx_comparisons_frame ON model_comparisons(frame_id);
CREATE INDEX idx_comparisons_model ON model_comparisons(model_name);
```

### 5.3 状态机联动

```
frames.annotation_status    annotations.review_status    联动规则
────────────────────────────────────────────────────────────────────
pending                     （无记录）                    初始态
annotating                  （写入中）                    事务开始
auto_checked                pending                      sampling.py 决策后
  → approved                approved                     未被抽中且 confidence >= 阈值
  → needs_review            needs_review                 被抽中或 confidence < 阈值
    → reviewed              reviewed                     人工审核完成
  → rejected                rejected                     严重格式错误
    → pending               （记录保留）                  rejected 帧回退为 pending 重新标注
exported                    （不变）                      导出完成
```

事务保证：写入 `annotations` 行和更新 `frames.annotation_status` 在同一个 SQLite 事务内。

### 5.4 缓存机制

`UNIQUE(frame_id, model_name, prompt_hash)` 即缓存键。`prompt_hash = sha256(system_prompt 原文 + user_prompt Jinja2 模板原文 + schema_version)`。注意是**模板原文**而非渲染后的内容——few-shot 示例变更会改变渲染结果但不改变模板，此时不触发重新标注（few-shot 微调不影响已有标注质量）。schema_version 或 prompt 模板变更时产生新 hash，自动重新打标。

---

## 6. Provider 抽象与注册

### 6.1 BaseProvider

```python
# pet_schema.render_prompt() 返回 tuple[str, str]，定义类型别名便于可读性
PromptPair = tuple[str, str]  # (system_prompt, user_prompt)

class BaseProvider(ABC):
    """所有 API Provider 的统一接口。"""

    @abstractmethod
    async def annotate(self, image_path: str, prompt: PromptPair,
                       api_key: str) -> ProviderResult:
        """发送单帧标注请求。"""

    @abstractmethod
    def supports_batch(self) -> bool:
        """是否支持原生 batch API。"""
```

`PromptPair` 是 `tuple[str, str]` 的类型别名，与 `pet_schema.render_prompt()` 返回类型一致。

`ProviderResult` dataclass：`raw_response`, `prompt_tokens`, `completion_tokens`, `latency_ms`。

`VLLMProvider` 的 `key_env` 为空字符串时，Provider 跳过 Authorization header。

### 6.2 Provider 实现

| Provider | 协议 | 图片传入 | 覆盖模型 |
|---|---|---|---|
| `OpenAICompatProvider` | OpenAI Chat Completions | base64 in message | 通义千问系列 |
| `DoubaoProvider` | 火山引擎 API | base64 或 URL | 豆包系列 |
| `VLLMProvider` | OpenAI 兼容 | base64 | 自建任意模型 |

新增模型：实现 `BaseProvider` 子类 + params.yaml 配置。协议兼容 OpenAI 的模型直接复用 `OpenAICompatProvider`。

### 6.3 ProviderRegistry

根据 params.yaml 初始化所有 Provider 和 RateTracker，提供 `get_primary()` 和 `get_all()` 方法。

### 6.4 RateTracker

```python
class RateTracker:
    """追踪单个模型下所有 account 的 RPM/TPM 用量。"""

    async def acquire(self, estimated_tokens: int = 0) -> str:
        """返回余量最多的 key。所有 key 满载时 await 等待。"""

    def record(self, key: str, tokens_used: int):
        """记录实际消耗。"""
```

实现：每个 key 维护 `collections.deque` 滑动窗口（60 秒），`acquire` 时计算 `当前用量/限额` 比值，选最低者。

---

## 7. Orchestrator 编排引擎

### 7.1 主循环

```python
class AnnotationOrchestrator:
    async def run(self, batch_size: int = 16):
        """主入口：批量拉 pending 帧，逐批处理。"""
        while True:
            frames = await self._fetch_pending(limit=batch_size)
            if not frames:
                break
            await self._process_batch(frames)
```

### 7.2 批处理流程

1. 事务：批量更新 `frames.annotation_status = 'annotating'`
2. 并发：`asyncio.gather` 所有帧的 `_process_single_frame`
3. 每帧内：`asyncio.gather` 所有配置模型的 `_annotate_one`
4. 事务：成功帧更新为 `auto_checked`，失败帧回退为 `pending`

### 7.3 并发控制

```
asyncio event loop
    │
    ├── frame_001 ─── qwen72b ──┐
    │                  doubao ───┤  Semaphore(50) 控制总并发
    ├── frame_002 ─── qwen72b ──┤
    │                  doubao ───┤
    ...                         │
                    RateTracker 选 key
```

### 7.4 SQLite 线程安全

```python
self._db_executor = ThreadPoolExecutor(max_workers=1)
# 所有数据库操作通过 run_in_executor 卸载
```

单 worker 理由：SQLite 写操作本身串行，多线程只增加锁竞争。

### 7.5 断点续跑保证

1. **帧级幂等**：`UNIQUE(frame_id, model_name, prompt_hash)` 缓存键
2. **批级持久化**：每 batch 完成后写回状态
3. **异常恢复**：启动时 `UPDATE frames SET annotation_status = 'pending' WHERE annotation_status = 'annotating'`

---

## 8. 质检模块

### 8.1 auto_check.py

- 从 `annotations` 表读主模型结果，`schema_valid` 和 `validation_errors` 已在打标时写入
- 额外：对比主模型与旁路模型的 `action.primary` 一致性，不一致写 warning

### 8.2 llm_judge.py

随机抽取已标注帧，用另一个 LLM 独立打标，对比一致性。复用 `ProviderRegistry`。

### 8.3 sampling.py

```python
def decide_review(annotation, sampling_rate, threshold) -> str:
    if not annotation.schema_valid:      return "needs_review"  # 验证失败必审
    if annotation.confidence_overall < threshold: return "needs_review"  # 低置信度必审
    if random.random() < sampling_rate:  return "needs_review"  # 随机抽样
    return "approved"
```

---

## 9. 人工审核模块

### 9.1 import_to_ls.py

- 查询 `review_status = 'needs_review'` 的记录
- 构建 Label Studio task，预填 VLM 输出作为 predictions
- 旁路模型结果附带在 task metadata 中供参考

### 9.2 export_from_ls.py

- 拉取已完成 annotation，更新 `review_status → reviewed`，同步更新 `frames.annotation_status`
- 审核员修改的输出覆盖 `annotations.parsed_output`

---

## 10. DPO 模块

### 10.1 配对策略（优先级从高到低）

1. **用户纠错**：`import_app_feedback.py` 生成的 chosen(人工) vs rejected(模型)
2. **跨模型配对**：主模型(approved) vs 旁路模型(同帧)，仅当主模型 confidence > 旁路模型时生成
3. **教师-学生配对**：72B(approved) vs 2B 推理结果

### 10.2 validate_pairs.py

五条校验规则（来自 DEVELOPMENT_GUIDE，不修改）：
1. chosen 通过 schema 验证
2. rejected 通过 schema 验证
3. chosen 和 rejected 的 narrative 不完全相同
4. 用户纠错 pair：rejected 有 inference_id 追踪
5. chosen.confidence_overall >= rejected.confidence_overall

---

## 11. 导出模块

### 11.1 to_sharegpt.py

查询主模型 `review_status IN ('approved', 'reviewed')` 的记录，用 `pet_schema.render_prompt()` 重建完整 prompt，输出 sharegpt JSONL。

### 11.2 to_dpo_pairs.py

从 `validate_pairs.py` 验证通过的配对中导出 DPO JSONL。

### 11.3 to_audio_labels.py

音频标签导出，与多模型无关。

---

## 12. 错误处理

### 12.1 API 调用层

- tenacity 重试：3 次，指数退避，仅对 `aiohttp.ClientError` 和 `asyncio.TimeoutError` 重试
- 429：重试 + RateTracker 标记 key 满载
- 4xx 其他：不重试，记录错误
- 5xx/超时：重试 3 次后跳过

### 12.2 Orchestrator 层

- 单帧失败不影响同 batch 其他帧（`return_exceptions=True`）
- CostTracker 超限：优雅停止
- SIGINT/SIGTERM：等当前 batch 完成后退出

### 12.3 数据库层

- 所有写操作在事务内，异常自动回滚
- 启动恢复：`annotating` 状态回退为 `pending`

---

## 13. DVC Pipeline

```yaml
stages:
  annotate:
    cmd: python -m pet_annotation.teacher.orchestrator
    deps:
      - src/pet_annotation/teacher/
      - params.yaml
    params:
      - annotation
      - models
    outs:
      - reports/annotation_stats.json

  quality_check:
    cmd: python -m pet_annotation.quality.auto_check
    deps:
      - src/pet_annotation/quality/
      - reports/annotation_stats.json     # 依赖 annotate 阶段输出
    params:
      - annotation.review_sampling_rate
      - annotation.low_confidence_threshold

  generate_pairs:
    cmd: python -m pet_annotation.dpo.generate_pairs
    deps:
      - src/pet_annotation/dpo/generate_pairs.py
      - src/pet_annotation/dpo/validate_pairs.py
    params:
      - dpo.min_pairs_per_release
    outs:
      - reports/dpo_pairs_stats.json

  export_sft:
    cmd: python -m pet_annotation.export.to_sharegpt
    deps:
      - src/pet_annotation/export/to_sharegpt.py
      - reports/annotation_stats.json     # 依赖 annotate 阶段
    outs:
      - exports/sft_train.jsonl

  export_dpo:
    cmd: python -m pet_annotation.export.to_dpo_pairs
    deps:
      - src/pet_annotation/export/to_dpo_pairs.py
      - reports/dpo_pairs_stats.json      # 依赖 generate_pairs 阶段
    outs:
      - exports/dpo_pairs.jsonl
```

---

## 14. 测试策略

| 测试文件 | 测试范围 | 方法 |
|---|---|---|
| `test_provider.py` | Provider 接口契约 | aioresponses mock HTTP |
| `test_rate_tracker.py` | 滑动窗口、key 选择、限额边界 | 单元测试 |
| `test_cache.py` | 缓存命中/未命中、prompt_hash 变更 | 内存 SQLite |
| `test_orchestrator.py` | 批量流程、断点续跑、异常恢复 | mock provider 注入 |
| `test_auto_check.py` | schema 验证通过/失败 | 样本数据 |
| `test_sampling.py` | 抽样率、低置信度强制审核 | 参数化测试 |
| `test_generate_pairs.py` | 三种配对策略 | 构造场景 |
| `test_validate_pairs.py` | 五条规则逐条正例+反例 | 参数化测试 |
| `test_export.py` | JSONL 格式正确性 | json.loads + schema 校验 |
| `test_store.py` | 表 CRUD、事务原子性 | 内存 SQLite |
| `test_config.py` | params.yaml 解析、环境变量 | 临时文件 |

---

## 15. CLI 入口

```python
@click.group()
def cli():
    """pet-annotation: VLM 打标、质检、审核、导出。"""

@cli.command()
def annotate(): ...    # 批量打标

@cli.command()
def check(): ...       # 质检

@cli.command()
def export(): ...      # 导出（--format sft|dpo|audio）

@cli.command()
def stats(): ...       # 进度统计
```

---

## 16. 依赖清单

依赖管理：`requirements.in` 定义松约束 → `pip-compile` 生成精确锁定的 `requirements.txt`。

```
pet-schema==1.0.0           # 固定 tag
aiohttp>=3.9,<4.0           # 异步 HTTP
tenacity>=8.0,<9.0          # 重试
click>=8.0,<9.0             # CLI
pydantic>=2.0,<3.0          # 配置
pyyaml>=6.0                 # params.yaml
label-studio-sdk>=1.0       # Label Studio API
dvc>=3.0                    # 管线
ruff                        # lint（dev）
mypy                        # type check（dev）
pytest>=7.0                 # test（dev）
pytest-asyncio>=0.21        # async test（dev）
aioresponses>=0.7           # mock（dev）
```

---

## 17. .env.example

```bash
# pet-annotation 环境变量模板
# 复制为 .env 并填入实际值，.env 已在 .gitignore 中

# Qwen / 通义千问 API Keys
QWEN_API_KEY_1=sk-xxx
QWEN_API_KEY_2=sk-xxx

# 豆包 / 火山引擎 API Keys
DOUBAO_API_KEY_1=xxx

# Label Studio
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=xxx
```
