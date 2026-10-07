import json
import pytest
from agent import get_weather   # 这是一个 StructuredTool 对象

def test_weather_city_not_found(monkeypatch):
    # 屏蔽高德配置，强制走 Open-Meteo 模拟路径
    monkeypatch.delenv("AMAP_KEY", raising=False)
    monkeypatch.delenv("WEATHER_PROXY", raising=False)
    def mock_get(*args, **kwargs):
        class MockResponse:
            def json(self):
                return {"results": []}
        return MockResponse()
    monkeypatch.setattr("requests.get", mock_get)
    # 使用 invoke 方法，传入字典参数
    result = json.loads(get_weather.invoke({"loc": "不存在的城市"}))
    assert "error" in result
    assert "未找到城市" in result["error"]

def test_weather_missing_current_weather(monkeypatch):
    monkeypatch.delenv("AMAP_KEY", raising=False)
    monkeypatch.delenv("WEATHER_PROXY", raising=False)
    call_count = 0
    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        class MockResponse:
            def json(self):
                if call_count == 1:
                    return {"results": [{"latitude": 39.9, "longitude": 116.4, "name": "北京"}]}
                else:
                    return {}  # 缺少 current_weather
        return MockResponse()
    monkeypatch.setattr("requests.get", mock_get)
    result = json.loads(get_weather.invoke({"loc": "北京"}))
    assert "error" in result
    assert "无法获取" in result["error"]

def test_weather_success(monkeypatch):
    monkeypatch.delenv("AMAP_KEY", raising=False)
    monkeypatch.delenv("WEATHER_PROXY", raising=False)
    call_count = 0
    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        class MockResponse:
            def json(self):
                if call_count == 1:
                    return {"results": [{"latitude": 39.9, "longitude": 116.4, "name": "北京"}]}
                else:
                    return {"current_weather": {"temperature": 20, "windspeed": 5, "weathercode": 0, "time": "2025-04-01T12:00"}}
        return MockResponse()
    monkeypatch.setattr("requests.get", mock_get)
    result = json.loads(get_weather.invoke({"loc": "北京"}))
    assert result["city"] == "北京"
    assert result["temperature_celsius"] == 20
    assert result["wind_speed_kmh"] == 5

def test_weather_exception(monkeypatch):
    monkeypatch.delenv("AMAP_KEY", raising=False)
    monkeypatch.delenv("WEATHER_PROXY", raising=False)
    def mock_get(*args, **kwargs):
        raise Exception("Network error")
    monkeypatch.setattr("requests.get", mock_get)
    result = json.loads(get_weather.invoke({"loc": "北京"}))
    assert "error" in result
    assert "Network error" in result["error"]

def test_weather_amap_success(monkeypatch):
    # 高德数据源：国内城市走 amap 地理编码(adcode) + 天气实况(lives)
    monkeypatch.setenv("AMAP_KEY", "test-key")
    monkeypatch.delenv("WEATHER_PROXY", raising=False)
    def mock_get(url, params=None, timeout=None, proxies=None):
        class MockResponse:
            def json(self):
                if "geocode/geo" in url:
                    return {"status": "1", "geocodes": [{"adcode": "350783", "city": "武夷山市"}]}
                return {"status": "1", "lives": [{"temperature": "13.0", "winddirection": "东北",
                                                    "windpower": "≤3", "weather": "晴",
                                                    "reporttime": "2026-10-09 10:00:00"}]}
        return MockResponse()
    monkeypatch.setattr("requests.get", mock_get)
    result = json.loads(get_weather.invoke({"loc": "武夷山"}))
    assert result["city"] == "武夷山市"
    assert result["temperature_celsius"] == 13.0
    assert result["source"] == "amap"
    assert "tea_frost_advice" in result

def test_weather_amap_forecast_tomorrow(monkeypatch):
    # 高德预报：date="明天" 时走 extensions=all，返回当日预报（温度范围/天气/风力）
    monkeypatch.setenv("AMAP_KEY", "test-key")
    monkeypatch.delenv("WEATHER_PROXY", raising=False)
    def mock_get(url, params=None, timeout=None, proxies=None):
        class MockResponse:
            def json(self):
                if "geocode/geo" in url:
                    return {"status": "1", "geocodes": [{"adcode": "350783", "city": "南平市"}]}
                if "all" in str(params):
                    return {"status": "1", "forecasts": [{"casts": [
                        {"date": "2026-10-08", "dayweather": "晴", "nightweather": "多云",
                         "daytemp": "22", "nighttemp": "12", "daywind": "北", "daypower": "1-3"},
                        {"date": "2026-10-09", "dayweather": "小雨", "nightweather": "小雨",
                         "daytemp": "18", "nighttemp": "10", "daywind": "东北", "daypower": "4-5"}]}]}
                return {"status": "0"}
        return MockResponse()
    monkeypatch.setattr("requests.get", mock_get)
    result = json.loads(get_weather.invoke({"loc": "武夷山", "date": "明天"}))
    assert result["type"] == "forecast"
    assert result["temp_min_celsius"] == 10.0
    assert result["temp_max_celsius"] == 18.0
    assert result["weather_day"] == "小雨"

def test_resolve_date_parsing():
    # 日期解析：自然语言/具体日期/非法输入/超远日期钳制
    from agent import _resolve_date
    off, _, label = _resolve_date("明天")
    assert off == 1 and label == "明天"
    off, _, label = _resolve_date("大后天")
    assert off == 3
    off, _, label = _resolve_date("随便写")
    assert off == 0 and label == "今天"
    off, _, _ = _resolve_date("2030-01-01")  # 超出7天窗口 → 钳制到最后一天
    assert off == 6