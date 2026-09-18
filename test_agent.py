import json
import pytest
from agent import get_weather   # 这是一个 StructuredTool 对象

def test_weather_city_not_found(monkeypatch):
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
    def mock_get(*args, **kwargs):
        raise Exception("Network error")
    monkeypatch.setattr("requests.get", mock_get)
    result = json.loads(get_weather.invoke({"loc": "北京"}))
    assert "error" in result
    assert "Network error" in result["error"]