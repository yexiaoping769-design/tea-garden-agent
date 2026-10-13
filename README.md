# tea-garden-agent 茶园智问

基于 LangGraph 的茶园种植领域智能助手：集成实时天气查询、联网搜索、本地文档阅读等工具，支持多轮会话与工具调用过程可视化。项目正在持续开发中。

> 背景：作者在参与武夷山茶园遥感图像标注项目期间，注意到茶农在农事决策（防霜冻、病虫害防治等）中缺少快速获取信息的工具，因此开发本助手。

## 核心指标速览

| 维度 | 结果 |
|---|---|
| **语料迭代** | AI 生成种子语料检索命中率 38% → 农业农村部官方标准语料 100%（70 知识块，4 份官方文档） |
| **检索链路** | BM25 关键词召回 + bge-small-zh 向量语义召回 → RRF 融合 → bge-reranker cross-encoder 重排（三阶段） |
| **检索评测** | 33 条知识库问题 RRF-only 命中 100%；+reranker 命中 100%，重排优化全部 33 条 top3 排序 |
| **端到端评测** | 50 条全场景评测，工具调度准确率 49/50（98%），LLM-as-Judge 通过率 48/50（96%），均分 2.78/3 |
| **评测发现** | LLM 裁判存在知识截止带来的误判（如 NY/T 4803-2025 新标准），优化裁判提示词并增加人工复核 |

## 架构流程图

```mermaid
flowchart TB
    subgraph 用户层
        A[浏览器前端]
    end

    subgraph 服务层
        B[Flask 后端 web_app.py]
        B1[/chat 对话接口]
        B2[/upload 上传入库接口]
    end

    subgraph Agent层
        C[LangGraph Agent]
        C1[ChatDeepSeek 大模型对话]
        C2[MemorySaver 多轮记忆]
        C3{工具选择}
    end

    subgraph 工具层
        D[get_weather 天气查询]
        E[TavilySearch 联网搜索]
        F[rag_search 知识检索]
    end

    subgraph RAG检索三阶段
        F1[BM25 关键词召回]
        F2[bge-small-zh 向量召回]
        F3[RRF 融合取 top-8]
        F4[bge-reranker cross-encoder 重排取 top-3]
    end

    subgraph 外部服务
        G[DeepSeek API]
        H[高德 + Open-Meteo 天气]
        I[Tavily API 联网搜索]
        J[Chroma 向量库 + bge-reranker 重排模型]
    end

    A -->|POST /chat| B
    A -->|POST /upload| B
    B --> C
    C --> C3
    C3 --> D
    C3 --> E
    C3 --> F
    C --> C1
    C1 --> G

    D --> H
    E --> I

    F --> F1
    F --> F2
    F1 --> F3
    F2 --> F3
    F3 --> F4
    F4 --> J
```

## 当前功能

- LangGraph Agent：基于 DeepSeek 的工具调用 Agent，支持多轮会话记忆（MemorySaver）
- 天气查询：高德 + Open-Meteo 双数据源，支持实时天气与未来 7 天预报，含茶园霜冻建议（详见下文「天气工具：双数据源策略」）
- 联网搜索：Tavily Search
- 三阶段混合检索：BM25 关键词（jieba 分词）+ 向量语义（bge-small-zh）→ RRF 融合 → bge-reranker cross-encoder 重排
- 本地文档阅读：支持 .txt / .docx / .pdf / .xlsx / .md
- Web 界面：Flask 后端 + 前端聊天页面，工具调用过程可视化，Markdown 渲染（marked + DOMPurify 防注入）

## 天气工具：双数据源策略

天气查询按顺序尝试两个数据源，失败自动降级，对用户透明：

1. **高德（主源，国内城市）**：查询日期在今天~大后天（0~3 天）时优先尝试。先经高德地理编码拿到城市 adcode，实况走 `extensions=base`，预报走 `extensions=all`（高德提供今起 4 天）。国外城市解析不出 adcode，自然落到下一通道。
2. **Open-Meteo（兜底，全球覆盖）**：国外城市、未配置 `AMAP_KEY`、高德请求失败，以及未来 4~6 天预报（超出高德预报窗口）时使用。

可用性设计：

- 所有 HTTP 请求经统一的重试函数：最多重试 3 次、单次超时 10 秒，吸收瞬时网络抖动
- Open-Meteo 通道在国内直连不稳定：直连优先，全部失败且配置了 `WEATHER_PROXY` 时自动改走本地代理

数据时效透明化（源于真实使用反馈：发现"实时"数据与查询时间存在十几分钟偏差）：

- 实况结果标注 `observation_time`（观测时间）、`query_time`（查询时间）、`data_age_minutes`（数据年龄）
- 观测超过 30 分钟时附加 `freshness_note`，说明公共气象源一般每 15~30 分钟更新一次——不谎称"实时"

双源输出归一：无论命中哪个数据源，返回统一的 JSON 结构（温度 / 天气 / 风力 / 茶园霜冻建议）；霜冻建议函数同时兼容高德天气文字与 Open-Meteo 天气代码两种输入。

## 开发计划（Roadmap）

### 已完成

- [x] RAG 知识检索：茶园种植手册（PDF）解析、切分、向量化、检索问答（农业农村部官方语料，70 知识块）
- [x] 三阶段混合检索：BM25 关键词召回（jieba 分词）+ bge-small-zh 向量语义召回 → RRF 融合 → bge-reranker cross-encoder 重排——评测驱动引入：50 条评测集暴露 3 个"专有名词精确匹配"型 badcase（单路向量 91%），BM25+RRF 混合检索修复至 100%；reranker 进一步优化 top3 排序质量（33/33 排序全部调整）
- [x] 领域工具：天气查询（高德 + Open-Meteo 双数据源，7 天预报，数据时效标注）+ 茶园霜冻预警判断
- [x] 评测体系：50 条领域评测集 + 三模式评测脚本 + 评测报告（检索两层迭代：语料升级 38%→100%，混合检索 91%→100%；端到端评测：工具调度 49/50，LLM 裁判通过率 48/50、均分 2.78/3，明细见 eval_results_e2e.json）
- [x] 安全加固：密钥统一环境变量管理（python-dotenv），移除硬编码敏感工具
- [x] Web 界面：Markdown 渲染（marked + DOMPurify 防注入）、工具调用过程可折叠面板展示
- [x] 在线上传入库：PDF/Word/txt/md 网页上传自动解析入知识库

### 未来计划

- [ ] LangSmith / OpenTelemetry 全链路观测（当前为单用户 demo，多用户化时引入）
- [ ] 多用户鉴权与会话隔离
- [ ] 更多领域工具：病虫害发生期预判、采摘窗口期建议
- [ ] 微气象站数据接入（武夷山高海拔茶园微气候差异大）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量：在项目根目录创建 .env 文件
DEEPSEEK_API_KEY=你的key
TAVILY_API_KEY=你的key
AMAP_KEY=你的key        # 可选：高德天气 API Key，国内城市优先走高德源
WEATHER_PROXY=http://127.0.0.1:12450  # 可选：Open-Meteo 兜底通道的本地代理

# 3. 运行测试
pytest test_agent.py -v

# 4. 启动 Web 服务
python web_app.py
# 浏览器打开 http://127.0.0.1:5050
```

## 技术栈

Python 3.12 / LangChain / LangGraph / DeepSeek API / Pydantic / Chroma / sentence-transformers（bge-small-zh + bge-reranker）/ rank_bm25 + jieba / Flask / pytest

## 项目踩坑与收获

**1. RAG 效果的上限由语料质量决定，不是模型大小**
种子期用 AI 生成的茶园文档做知识库，检索命中率只有 38%。换成农业农村部 4 份官方标准（NY/T 4803-2025 等），不换模型、不换切分参数，直接跳到 100%。语料是 RAG 的天花板——垃圾语料再强的检索也救不回来。

**2. 向量检索专治语义泛化，BM25 专治专有名词精确匹配**
"蓑蛾""出口用药"这类专有名词，bi-encoder 按语义相似性排序容易把它们推到候选集后面（因为没见过太多蓑蛾相关的语义变体）。BM25 是词频匹配，对精确词匹配更敏感。双路 + RRF 融合是低成本高收益的组合。

**3. RRF 之后为什么还要 reranker**
RRF 是名次融合（late fusion），只看召回位次，不看 query-doc 内容交互。cross-encoder reranker 把 query 和 doc 拼一起编码，交互更充分，但速度慢。所以只在 RRF 筛出来的 top-8 小候选集上跑 reranker，取 top-3。评测显示命中率没变化（RRF-only 已经 100%），但 reranker 调整了全部 33 条题的 top3 排序——LLM 只看 top-k，首位来源权重最大。

**4. LLM-as-Judge 不能完全盲信**
评测用 DeepSeek 当裁判，发现两类误判：① 知识库包含 2025 年发布的 NY/T 4803-2025 新标准，但裁判训练数据截止在 2024 年之前，把正确的新标准回答判成"编造"；② 诚实性测试题（"有机茶认证是什么"），Agent 主动联网搜到了真实法规，裁判按"应如实说不知道"判 0 分。应对措施：裁判提示词里声明知识库覆盖的标准年份，人工复核边界 case。

**5. 工具出错不抛异常，返回错误 JSON 让 Agent 感知**
天气接口超时、高德返回空数据、JSON 解析失败——这些场景如果直接 raise，Agent 流程会卡死（用户看到"正在处理"然后什么都没有）。改成返回 `{"error": "高德天气服务暂时不可用，请稍后再试"}` 的 JSON，模型能把错误翻译成用户友好的提示，Agent 还能继续处理其他问题。

**6. PDF 提取的 Unicode 私用区字符需要清洗**
官方 PDF 用了自定义字体映射，章节号中的小数点（"．"）被映射到私用区码位（U+E000 等），表现为 `5􀆰2􀆰4` 这种乱码。解决方案：NFKC 归一化把全角转半角，然后正则匹配"数字+私用区+数字"替换成"."，其余无上下文依据的私用区字符直接剔除——避免污染向量编码。
