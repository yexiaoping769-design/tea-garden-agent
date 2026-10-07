# tea-garden-agent 茶园智问

基于 LangGraph 的茶园种植领域智能助手：集成实时天气查询、联网搜索、本地文档阅读等工具，支持多轮会话与工具调用过程可视化。项目正在持续开发中。

> 背景：作者在参与武夷山茶园遥感图像标注项目期间，注意到茶农在农事决策（防霜冻、病虫害防治等）中缺少快速获取信息的工具，因此开发本助手。

## 当前功能

- LangGraph Agent：基于 DeepSeek 的工具调用 Agent，支持多轮会话记忆（MemorySaver）
- 天气查询：高德 + Open-Meteo 双数据源，支持实时天气与未来 7 天预报，含茶园霜冻建议（详见下文「天气工具：双数据源策略」）
- 联网搜索：Tavily Search
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

- [ ] RAG 知识检索：茶园种植手册（PDF）解析、切分、向量化、检索问答
- [ ] 领域工具：天气 + 霜冻预警判断
- [ ] 评测体系：自建领域评测集 + 自动化评测脚本 + 评测报告
- [ ] 安全加固：移除敏感工具、密钥管理规范化

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量：在项目根目录创建 .env 文件
DEEPSEEK_API_KEY=你的key
TAVILY_API_KEY=你的key

# 3. 运行测试
pytest test_agent.py -v

# 4. 启动 Web 服务
python web_app.py
# 浏览器打开 http://127.0.0.1:5050
```

## 技术栈

Python 3.12 / LangChain / LangGraph / DeepSeek API / Pydantic / pytest / Flask
