# tea-garden-agent 茶园智问

基于 LangGraph 的茶园种植领域智能助手：集成实时天气查询、联网搜索、本地文档阅读等工具，支持多轮会话与工具调用过程可视化。项目正在持续开发中。

> 背景：作者在参与武夷山茶园遥感图像标注项目期间，注意到茶农在农事决策（防霜冻、病虫害防治等）中缺少快速获取信息的工具，因此开发本助手。

## 当前功能

- LangGraph Agent：基于 DeepSeek 的工具调用 Agent，支持多轮会话记忆（MemorySaver）
- 实时天气查询：Open-Meteo API，含地理编码 + 参数校验（Pydantic）
- 联网搜索：Tavily Search
- 本地文档阅读：支持 .txt / .docx / .pdf / .xlsx
- Web 界面：Flask 后端 + 前端聊天页面，工具调用过程可视化

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
