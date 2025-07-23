from flask import Flask, request, jsonify
import time
import random

app = Flask(__name__)

buffer = []
threshold = 5  # Simulate training after 5 observations

def fake_training():
    print("Training... blocking for 5 seconds")
    time.sleep(5)  # Simulate training
    print("Training complete")

@app.route("/get_next_action", methods=["POST"])
def get_next_action():
    data = request.get_json()
    buffer.append(data)

    if len(buffer) >= threshold:
        fake_training()
        buffer.clear()

    return jsonify({"action": random.choice(["up", "down", "left", "right"])})

if __name__ == "__main__":
    app.run(port=5000)
