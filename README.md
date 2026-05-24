# Deep Research Agent

基于 LangGraph 的深度研究 Agent：搜索 → 生成 → 反思 → 迭代，产出带引用的高质量研究报告。

## 工作流程

```text
用户问题 → 生成搜索Query → 搜索网络(并行) → 生成答案 → 反思评估
                ↑                                          |
                └────────── 不满意 / 未达上限 ←─────────────┘
```

| 节点 | 说明 | 模型 |
| --- | --- | --- |
| Generate Query | 将问题拆解为 2-3 个搜索关键词 | `deepseek-v4-flash` |
| Search | 并行调用 Serper (Google) 搜索，收集结果 | — |
| Generate Answer | 基于搜索结果生成带引用的答案 | `deepseek-v4-pro` |
| Reflect | 自评答案质量，不通过则带回反馈重新搜索 | `deepseek-v4-flash` |

- 搜索使用 `ThreadPoolExecutor` 并行请求，多个 query 同时发出
- 最多迭代 3 轮，每轮耗时约 5-10s
- 运行时每个节点打印进度和耗时，方便定位瓶颈

## 快速开始

### 1. 安装依赖

```bash
pip install langgraph langchain-openai requests python-dotenv
```

### 2. 配置 API Key

申请以下两个 Key：

- [DeepSeek](https://platform.deepseek.com) — LLM 调用
- [Serper](https://serper.dev) — Google 搜索（免费额度 2500 次/月）

写入 `.env` 文件：

```ini
DEEPSEEK_API_KEY=your-deepseek-key
SERPER_API_KEY=your-serper-key
```

### 3. 运行

```bash
python test.py
```

修改 `main` 中的 `question` 即可研究任意问题。

## 项目结构

```text
langgraph/
├── test.py      # 主程序
├── .env         # API Key（不提交）
├── .gitignore
└── README.md
```
