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
        # 关键修复点：确保所有输出都为 ASCII 安全形式
        return json.dumps(result, ensure_ascii=True)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=True)
@tool
def get_env_var(var_name: str) -> str:
    """获取系统环境变量的值。"""
    value = os.environ.get(var_name)
    if value is None:
        return f"环境变量 {var_name} 不存在"
    if len(value) > 500:
        value = value[:500] + "...(已截断)"
    return value

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

tools=[search_tool,get_weather,get_env_var,read_local_file]
prompt="""
    你是一名乐于助人的智能助手，擅长根据用户的问题选择合适的工具来查询信息并回答。

    当用户的问题涉及天气信息时，你应优先调用"get_weather"工具，查询用户指定城市的实时天气，并在回答中总结查询结果。

    当用户的问题涉及"新闻、事件、实时动态"时，你应优先调用"search_tool"工具，检索相关的最新信息，并在回答中简要概述。

    当用户的问题涉及“环境变量、系统信息、我的用户名、PATH”等内容时，调用“get_env_var”工具。

    当用户要求“读取本地文件”（例如“帮我读一下D:\test.txt”）时，调用“read_local_file”工具。

    如果问题同时包含多个方面，依次调用对应的工具，合并结果后回复。

    所有回答应使用简体中文，条理清晰、简洁友好。
"""

#创建图
agent = create_agent(model=model,
                    system_prompt=prompt,
                    tools=tools,
                    checkpointer=MemorySaver()
                    )

