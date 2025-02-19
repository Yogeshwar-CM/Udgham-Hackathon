from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/api', methods=['POST'])
def api():
    data = request.json
    response = {"message": f"Received: {data}"}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)