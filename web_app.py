import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from agent import agent
from rag_core import ingest_file

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# 上传文件暂存目录 + 允许的格式
UPLOAD_DIR = Path(__file__).parent / "uploads"
ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}

# 用于记录每个会话已处理的消息数量（用于调试）
processed_counts = {}

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """在线上传文档入知识库：保存 → 解析 → 切分 → 向量化 → 增量写入 Chroma。"""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "没有收到文件"}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"不支持的文件类型 {ext}，仅支持 PDF/Word(.docx)/txt/md"}), 400

    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    try:
        chunks = ingest_file(save_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"入库失败：{e}"}), 500
    return jsonify({"chunks": chunks, "filename": file.filename})

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