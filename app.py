from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_text():
    keyword = request.form['keyword']
    location = request.form['location']
    processed_text = f"You entered: {keyword} and {location}"
    return jsonify(result=processed_text)

if __name__ == "__main__":
    app.run(debug=True)