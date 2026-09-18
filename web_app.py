import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from agent import agent

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# 用于记录每个会话已处理的消息数量（用于调试）
processed_counts = {}

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "消息不能为空"}), 400

    thread_id = data.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # 调用 agent，获取所有消息
        result = agent.invoke({"messages": [("user", user_msg)]}, config=config)
        all_messages = result["messages"]

        # 获取本次会话之前已有的消息数量（如果未记录，则初始为0）
        previous_count = processed_counts.get(thread_id, 0)
        # 本轮新增的消息 = all_messages[previous_count:] 
        new_messages = all_messages[previous_count:]
        # 更新已处理数量
        processed_counts[thread_id] = len(all_messages)

        # 仅从新增消息中提取工具调用信息
        tool_calls_info = []
        for msg in new_messages:
            # AI 消息中的工具调用请求
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_info.append({
                        "type": "call",
                        "tool": tc.get("name"),
                        "args": tc.get("args")
                    })
            # 工具返回消息（ToolMessage）
            if hasattr(msg, 'type') and msg.type == 'tool':
                tool_calls_info.append({
                    "type": "result",
                    "content": msg.content[:200]
                })

        # 最终回答（最后一条 AI 消息）
        reply = all_messages[-1].content

        return jsonify({
            "reply": reply,
            "tool_calls_info": tool_calls_info,
            "thread_id": thread_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)