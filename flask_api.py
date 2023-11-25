from flask import Flask, request, jsonify
from flask_cors import CORS  
from chatbot_response import *


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.route('/api/search/smart-agent/search/<term>', methods=['GET', 'POST'])
def test(term):
    chatbot_message=get_response(term)
    return jsonify(chatbot_message)
app.run(debug=True, port=9090)