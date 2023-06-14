from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from model import DQNAgent

app = Flask(__name__)

@app.route('/recommend', methods=['get'])
def recommend_content():
    agent=DQNAgent()
    action=agent.act()
    response={
        'module':action.item()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
