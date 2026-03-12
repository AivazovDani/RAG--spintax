from flask import Flask, render_template, request
from history_aware_generation import ask_question
import os
from waitress import serve

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run', methods = ['POST', 'GET'])
def run_script():
    value1 = request.form.get('Field1')
    answer = ask_question(value1)
    return answer

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    serve(app, host="0.0.0.0", port=port)

