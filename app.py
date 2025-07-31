from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from query_answering import get_answer

app = Flask(__name__)
app.secret_key = 'Bearer 504635995f102f61f796b6b872fc2c4aab9746b70aed2ccac264c0928c881adc'

# 🔹 Web UI route
@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        query = request.form["query"]
        response_data = get_answer(query)

        # Keep JSON clean
        filtered_response = {
            "answer": response_data.get("answer", "")
        }

        session["chat_history"].append({
            "user": query,
            "json": filtered_response
        })
        session.modified = True

    return render_template("index.html", chat_history=session.get("chat_history", []))

# 🔹 Clear chat history
@app.route("/reset", methods=["GET"])
def reset():
    session.pop("chat_history", None)
    return redirect(url_for('index'))

# ✅ HackRx-required public API endpoint
@app.route("/api/v1/hackrx/run", methods=["POST"])
def hackrx_api():
    try:
        data = request.get_json()
        query = data.get("query") or data.get("documents")

        if not query:
            return jsonify({"error": "Missing 'query' or 'documents' field in request."}), 400

        answer = get_answer(query)
        return jsonify(answer)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
