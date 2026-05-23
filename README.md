# Deep Research Agent

基于 LangGraph 的深度研究 Agent，自动搜索、反思、迭代，生成高质量研究报告。

## 工作流程

```text
用户问题 → 生成搜索Query → 搜索网络 → 生成答案 → 反思评估
                ↑                                    |
                └──────── 不满意 / 未达上限 ←────────┘
```

1. **Generate Query** — LLM 将问题拆解为 2-3 个精准搜索关键词
2. **Search** — 通过 DuckDuckGo 执行搜索，收集结果
3. **Generate Answer** — LLM 基于搜索结果生成带引用的答案
4. **Reflect** — LLM 自我评审，判断答案是否充分；不满足则带着反馈回到第 1 步

最多迭代 3 轮，确保答案质量。

## 快速开始

### 1. 安装依赖

```bash
pip install langgraph langchain-openai duckduckgo-search python-dotenv
```

### 2. 配置 API Key

在 `.env` 文件中设置 DeepSeek API Key：

```ini
DEEPSEEK_API_KEY=your-key-here
```

### 3. 运行

```bash
python test.py
```

修改 `main` 中的 `question` 即可研究任意问题。

## 项目结构

```
langgraph/
├── test.py      # 主程序
├── .env         # API Key（不提交）
├── .gitignore
└── README.md
```
