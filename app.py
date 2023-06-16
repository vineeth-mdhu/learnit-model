from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from model import DQNAgent

app = Flask(__name__)

@app.route('/recommend', methods=['get'])
def recommend_content():
    args=request.args.to_dict()
    print(args)
    student_id=args.get('student_id')
    course_id=args.get('course_id')
    agent=DQNAgent(student_id,course_id)
    action=agent.act()
    response={
        'module':action.item()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
