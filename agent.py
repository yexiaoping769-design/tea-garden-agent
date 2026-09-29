import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from pydantic import BaseModel,Field
from langgraph.checkpoint.memory import MemorySaver
import requests,json
from rag_core import search_tea_knowledge as rag_search


#加载环境变量
load_dotenv(override=True)

#内置搜索工具
search_tool = TavilySearch(max_results = 5,topic = "general")

class WeatherQuery(BaseModel):
    loc:str = Field(description="The location name of the city")

@tool(args_schema=WeatherQuery)
def get_weather(loc: str) -> str:
    """
    查询即时天气函数。
    当用户询问某个城市的天气情况时，可以使用此工具。
    :param loc: 城市名称,支持中文或英文,例如“北京”或“Beijing”。
    :return: 返回一个 JSON 字符串，包含温度、风速、天气代码和时间等信息。
    """
    try:
        # 第一步：通过城市名获取经纬度（地理编码）
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": loc,
            "count": 1,
            "language": "zh",
            "format": "json"
        }
        geo_resp = requests.get(geo_url, params=geo_params)
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            # 关键修复点：将错误信息中的中文也转义
            return json.dumps({"error": f"未找到城市 '{loc}'"}, ensure_ascii=True)

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        # 第二步：用经纬度获取实时天气
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "timezone": "auto"
        }
        weather_resp = requests.get(weather_url, params=weather_params)
        weather_data = weather_resp.json()

        if "current_weather" not in weather_data:
            return json.dumps({"error": f"无法获取 '{loc}' 的天气数据"}, ensure_ascii=True)

        current = weather_data["current_weather"]
        result = {
            "city": loc,
            "temperature_celsius": current.get("temperature"),
            "wind_speed_kmh": current.get("windspeed"),
            "weather_code": current.get("weathercode"),
            "observation_time": current.get("time")
        }

        # 茶园防霜预警：结合气温和天气代码给出农事建议
        temp = current.get("temperature")
        code = current.get("weathercode")
        if temp is not None and temp <= 2.0:
            result["tea_frost_advice"] = (
                "⚠️ 低温霜冻风险：当前气温接近冰点，茶芽易受冻。"
                "建议立即采取覆盖防霜、熏烟防霜或灌水防霜等措施。"
            )
        elif code in (71, 73, 75, 77, 85, 86, 66, 67):
            result["tea_frost_advice"] = "⚠️ 出现雨雪/冻雨天气，注意茶树防冻，及时清理芽叶积雪。"
        elif temp is not None and temp <= 8.0:
            result["tea_frost_advice"] = "气温偏低，注意倒春寒，关注夜间气温变化。"
        else:
            result["tea_frost_advice"] = "气温正常，暂无霜冻风险。"

        # 关键修复点：确保所有输出都为 ASCII 安全形式
        return json.dumps(result, ensure_ascii=True)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=True)

class TeaKnowledgeQuery(BaseModel):
    query: str = Field(description="茶园管理相关的检索问题，例如'茶饼病怎么防治'、'春茶前如何施肥'")

@tool(args_schema=TeaKnowledgeQuery)
def search_tea_knowledge(query: str) -> str:
    """
    检索本地茶园知识库（含农业农村部茶树病虫害防控、栽培技术标准等官方文档）。
    当用户询问茶树种植、病虫害防治、施肥、修剪、采摘、霜冻防御等茶园管理知识时，使用此工具。
    """
    return rag_search(query)

@tool
def read_local_file(file_path: str) -> str:
    """
    读取本地文件的内容（支持 .txt, .docx, .pdf, .xlsx）。
    返回文件内容的前2000个字符（或前20行表格数据），非文本文件会报错。
    """
    import os
    try:
        ext = os.path.splitext(file_path)[1].lower()
        content = ""

        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        elif ext == '.docx':
            from docx import Document
            doc = Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])

        elif ext == '.pdf':
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            content = "\n".join([page.extract_text() or "" for page in reader.pages])

        elif ext in ['.xlsx', '.xls']:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            sheet = wb.active
            rows = list(sheet.iter_rows(values_only=True))
            # 将单元格内容转为字符串，每行用制表符分隔
            content = "\n".join(["\t".join(str(cell) if cell is not None else "" for cell in row) for row in rows])
        else:
            return f"不支持的文件类型: {ext}，目前仅支持 .txt, .docx, .pdf, .xlsx"

        if not content.strip():
            return "文件内容为空或无法解析"

        if len(content) > 2000:
            content = content[:2000] + "\n...（内容过长，已截断）"
        return content

    except FileNotFoundError:
        return f"错误：文件 '{file_path}' 不存在"
    except Exception as e:
        return f"读取失败：{str(e)}"

# ================== 3. 创建 Agent 并运行 ==================
# 请将下面的字符串替换为你自己的 DeepSeek API Key

#创建模型
DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
model = ChatDeepSeek(model="deepseek-chat", api_key=DeepSeek_API_KEY)

tools=[search_tool,get_weather,search_tea_knowledge,read_local_file]
prompt="""
    你是"茶园智问"，一名专业的茶园种植管理助手，服务对象是茶农和茶园管理者。

    工具使用规则：
    1. 当用户询问天气、气温，或结合天气问农事（例如"今天要不要防霜冻"）时，调用"get_weather"查询实时天气，
       并结合返回结果中的 tea_frost_advice 字段给出农事建议。
    2. 当用户询问茶树病虫害防治、种植、施肥、修剪、采摘等茶园管理知识时，优先调用"search_tea_knowledge"
       检索本地知识库（内含农业农村部官方标准），回答时标注来源。
    3. 当用户询问最新新闻、茶叶市场行情等实时动态时，调用"search_tool"联网检索。
    4. 当用户要求读取本地文件时，调用"read_local_file"。
    5. 一个问题涉及多个方面时，依次调用对应的工具，综合结果后回答。

    回答要求：使用简体中文，条理清晰、简洁友好；引用知识库内容时注明来源；
    知识库和工具都无法确认的内容，如实告知，不要编造。
"""

#创建图
agent = create_agent(model=model,
                    system_prompt=prompt,
                    tools=tools,
                    checkpointer=MemorySaver()
                    )

