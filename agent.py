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
import requests,json,time
from datetime import datetime, timedelta
from rag_core import search_tea_knowledge as rag_search

#加载环境变量（必须先于 WEATHER_PROXY 读取）
load_dotenv(override=True)

def _fetch_with_retry(url, params=None, tries=3, timeout=10, proxies=None):
    """带自动重试的 HTTP GET：瞬时断连（如代理抖动）时最多重试 tries 次。"""
    last_err = None
    for i in range(tries):
        try:
            return requests.get(url, params=params, timeout=timeout, proxies=proxies)
        except Exception as e:
            last_err = e
            if i < tries - 1:
                time.sleep(1)  # 等 1 秒再试
    raise last_err

# 天气请求兜底代理：在 _fetch_weather 内部读取（便于测试时用环境变量控制）
def _fetch_weather(url, params):
    """双通道天气请求：直连优先（国内访问 open-meteo 不稳，两种通道各有失败模式），
    全部失败且 .env 配置了 WEATHER_PROXY 时，自动改走本地代理兜底。"""
    proxy_url = os.getenv("WEATHER_PROXY")
    try:
        return _fetch_with_retry(url, params)
    except Exception:
        if proxy_url:
            proxy = {"http": proxy_url, "https": proxy_url}
            return _fetch_with_retry(url, params, proxies=proxy)
        raise


def _frost_advice(temp, weather_text="", code=None):
    """茶园防霜建议：结合气温与天气现象（兼容高德文字 / open-meteo 代码两种输入）。"""
    snow_like = ("雪", "冻雨", "冰雹")
    if temp is not None and temp <= 2.0:
        return "⚠️ 低温霜冻风险：当前气温接近冰点，茶芽易受冻。建议立即采取覆盖防霜、熏烟防霜或灌水防霜等措施。"
    if (weather_text and any(k in weather_text for k in snow_like)) or (code in (71, 73, 75, 77, 85, 86, 66, 67)):
        return "⚠️ 出现雨雪/冻雨天气，注意茶树防冻，及时清理芽叶积雪。"
    if temp is not None and temp <= 8.0:
        return "气温偏低，注意倒春寒，关注夜间气温变化。"
    return "气温正常，暂无霜冻风险。"


#内置搜索工具
search_tool = TavilySearch(max_results = 5,topic = "general")

class WeatherQuery(BaseModel):
    loc: str = Field(description="The location name of the city")
    date: str = Field(default="今天",
                      description="要查询哪天的天气：'今天'(默认,实时观测)、'明天'、'后天'、'大后天'，或具体日期YYYY-MM-DD，最远支持未来6天")

def _resolve_date(date_str):
    """把'今天/明天/后天/大后天'或'YYYY-MM-DD'解析为 (偏移0-6天, ISO日期, 中文标签)。非法输入按今天处理。"""
    today = datetime.now().date()
    s = (date_str or "今天").strip()
    mapping = {"今天": 0, "今日": 0, "明天": 1, "明日": 1, "后天": 2, "大后天": 3}
    if s in mapping:
        offset = mapping[s]
    else:
        try:
            offset = (datetime.strptime(s, "%Y-%m-%d").date() - today).days
        except ValueError:
            offset = 0
    offset = max(0, min(offset, 6))  # 预报窗口：今天起共7天
    labels = {0: "今天", 1: "明天", 2: "后天", 3: "大后天"}
    return offset, (today + timedelta(days=offset)).isoformat(), labels.get(offset, f"{offset}天后")

def _weather_code_cn(code):
    """open-meteo 天气代码 → 中文描述。"""
    m = {0: "晴", 1: "大致晴朗", 2: "多云", 3: "阴", 45: "雾", 48: "雾凇",
         51: "毛毛雨", 53: "毛毛雨", 55: "浓毛毛雨",
         61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "强冻雨",
         71: "小雪", 73: "中雪", 75: "大雪", 77: "米雪",
         80: "阵雨", 81: "阵雨", 82: "强阵雨", 85: "阵雪", 86: "阵雪",
         95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "雷阵雨伴冰雹"}
    return m.get(code, f"天气代码{code}")

@tool(args_schema=WeatherQuery)
def get_weather(loc: str, date: str = "今天") -> str:
    """
    查询天气函数：支持实时天气与未来7天天气预报。
    当用户询问某个城市今天的天气时返回实时观测；询问明天/后天/未来日期时返回当日预报（含温度范围、天气、风力）。
    :param loc: 城市名称,支持中文或英文,例如“北京”或“Beijing”。
    :param date: 查询哪天的天气：'今天'(默认)、'明天'、'后天'，或日期YYYY-MM-DD。
    :return: 返回一个 JSON 字符串，实时天气含观测时间与数据时效，预报含温度范围与天气状况。
    """
    try:
        offset, target_date, date_label = _resolve_date(date)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ===== 数据源1：高德（国内城市；实况 + 未来3天预报）=====
        amap_key = os.getenv("AMAP_KEY")
        if amap_key and offset <= 3:
            try:
                geo = _fetch_with_retry(
                    "https://restapi.amap.com/v3/geocode/geo",
                    params={"address": loc, "key": amap_key},
                )
                g = geo.json()
                if g.get("status") == "1" and g.get("geocodes"):
                    geo0 = g["geocodes"][0]
                    adcode = geo0.get("adcode")
                    city_name = geo0.get("city") if isinstance(geo0.get("city"), str) and geo0.get("city") else loc
                    if adcode:
                        ext = "base" if offset == 0 else "all"
                        w = _fetch_with_retry(
                            "https://restapi.amap.com/v3/weather/weatherInfo",
                            params={"city": adcode, "key": amap_key, "extensions": ext},
                        )
                        wd = w.json()
                        if wd.get("status") == "1" and offset == 0 and wd.get("lives"):
                            # 实时天气（含数据时效标注）
                            live = wd["lives"][0]
                            try:
                                temp = float(live.get("temperature"))
                            except (TypeError, ValueError):
                                temp = None
                            result = {
                                "city": city_name,
                                "date_label": "今天",
                                "type": "realtime",
                                "temperature_celsius": temp,
                                "weather": live.get("weather"),
                                "wind_direction": live.get("winddirection"),
                                "wind_power": live.get("windpower"),
                                "observation_time": live.get("reporttime"),
                                "query_time": now_str,
                                "source": "amap",
                            }
                            try:
                                obs = datetime.strptime(live.get("reporttime"), "%Y-%m-%d %H:%M:%S")
                                age = int((datetime.now() - obs).total_seconds() // 60)
                            except (TypeError, ValueError):
                                age = None
                            result["data_age_minutes"] = age
                            if age is not None and age > 30:
                                result["freshness_note"] = f"实况为{age}分钟前观测值，公共气象源一般每15~30分钟更新一次"
                            result["tea_frost_advice"] = _frost_advice(temp, live.get("weather") or "")
                            return json.dumps(result, ensure_ascii=True)
                        if wd.get("status") == "1" and offset >= 1 and wd.get("forecasts"):
                            # 天气预报（高德提供今起4天，取第 offset 天）
                            casts = wd["forecasts"][0].get("casts", [])
                            if offset < len(casts):
                                c = casts[offset]
                                try:
                                    tmin = float(c.get("nighttemp"))
                                    tmax = float(c.get("daytemp"))
                                except (TypeError, ValueError):
                                    tmin = tmax = None
                                result = {
                                    "city": city_name,
                                    "date": target_date,
                                    "date_label": date_label,
                                    "type": "forecast",
                                    "weather_day": c.get("dayweather"),
                                    "weather_night": c.get("nightweather"),
                                    "temp_max_celsius": tmax,
                                    "temp_min_celsius": tmin,
                                    "wind_direction_day": c.get("daywind"),
                                    "wind_power_day": c.get("daypower"),
                                    "source": "amap",
                                }
                                day_night = (c.get("dayweather") or "") + (c.get("nightweather") or "")
                                result["tea_frost_advice"] = _frost_advice(tmin, day_night)
                                return json.dumps(result, ensure_ascii=True)
            except Exception:
                pass  # 高德通道失败时，降级走 Open-Meteo

        # ===== 数据源2：Open-Meteo（国外城市 / 高德未配置或失败 / 未来4-6天预报）=====
        # 第一步：通过城市名获取经纬度（地理编码）
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": loc,
            "count": 1,
            "language": "zh",
            "format": "json"
        }
        geo_resp = _fetch_weather(geo_url, geo_params)
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            # 关键修复点：将错误信息中的中文也转义
            return json.dumps({"error": f"未找到城市 '{loc}'"}, ensure_ascii=True)

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        if offset == 0:
            # 第二步：用经纬度获取实时天气
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
                "timezone": "auto"
            }
            weather_resp = _fetch_weather("https://api.open-meteo.com/v1/forecast", weather_params)
            weather_data = weather_resp.json()

            if "current_weather" not in weather_data:
                return json.dumps({"error": f"无法获取 '{loc}' 的天气数据"}, ensure_ascii=True)

            current = weather_data["current_weather"]
            temp = current.get("temperature")
            code = current.get("weathercode")
            result = {
                "city": loc,
                "date_label": "今天",
                "type": "realtime",
                "temperature_celsius": temp,
                "weather": _weather_code_cn(code),
                "wind_speed_kmh": current.get("windspeed"),
                "weather_code": code,
                "observation_time": current.get("time"),
                "query_time": now_str,
                "source": "open-meteo",
            }
            result["tea_frost_advice"] = _frost_advice(temp, code=code)
            return json.dumps(result, ensure_ascii=True)

        # 第二步（预报）：获取未来7天日级预报
        forecast_params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "forecast_days": 7,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,precipitation_sum",
        }
        f_resp = _fetch_weather("https://api.open-meteo.com/v1/forecast", forecast_params)
        f_data = f_resp.json()
        daily = f_data.get("daily", {})
        dates = daily.get("time", [])
        if target_date not in dates:
            return json.dumps({"error": f"无法获取 '{loc}' 在 {target_date} 的预报"}, ensure_ascii=True)
        i = dates.index(target_date)
        tmin = daily.get("temperature_2m_min", [None])[i]
        tmax = daily.get("temperature_2m_max", [None])[i]
        code = daily.get("weather_code", [None])[i]
        result = {
            "city": loc,
            "date": target_date,
            "date_label": date_label,
            "type": "forecast",
            "weather_day": _weather_code_cn(code),
            "temp_max_celsius": tmax,
            "temp_min_celsius": tmin,
            "wind_speed_max_kmh": daily.get("wind_speed_10m_max", [None])[i],
            "precipitation_sum_mm": daily.get("precipitation_sum", [None])[i],
            "source": "open-meteo",
        }
        result["tea_frost_advice"] = _frost_advice(tmin, code=code)
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
    读取本地文件的内容（支持 .txt, .md, .docx, .pdf, .xlsx）。
    返回文件内容的前2000个字符（或前20行表格数据），非文本文件会报错。
    """
    import os
    try:
        ext = os.path.splitext(file_path)[1].lower()
        content = ""

        if ext in ('.txt', '.md'):
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
            return f"不支持的文件类型: {ext}，目前仅支持 .txt, .md, .docx, .pdf, .xlsx"

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
    1. 当用户询问天气、气温，或结合天气问农事（例如"今天要不要防霜冻"）时，调用"get_weather"查询。
       该工具支持实时天气和未来7天预报：问今天传 date="今天"（返回实时观测）；
       问"明天/后天/大后天"或具体日期时传对应 date（返回当日预报，含温度范围、天气、风力）。
       回答时必须标注观测时间或预报日期；若返回结果含 freshness_note 字段，说明数据存在延迟，请如实告知用户。
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

