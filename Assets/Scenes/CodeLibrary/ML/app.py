from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import random
import threading
from model import Model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global thread lock to make the entire application single-threaded
_model_lock = threading.Lock()

model = Model(state_dim=6, action_dim=4)

@app.route("/get_next_action", methods=["POST"])
def get_next_action():
    # Global lock makes this entire endpoint single-threaded
    with _model_lock:
        data = request.get_json()
        state = data["state"]
        reward = data["reward"]

        action = model.predict(state, reward)

        return jsonify({"action": action})

if __name__ == "__main__":
    app.run(port=5001)
