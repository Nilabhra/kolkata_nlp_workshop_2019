from sklearn.externals import joblib
from flask import Flask, request, jsonify
from string import digits


app = Flask(__name__)


def remove_digits(s: str) -> str:
    remove_digits = str.maketrans('', '', digits)
    res = s.translate(remove_digits)
    return res


@app.route('/infer', methods=['POST'])
def infer():
    global model
    text = request.json['text']
    text = remove_digits(text)
    pred = model.predict([text])
    response = {'sentiment': pred[0]}
    return jsonify(response)


if __name__ == '__main__':
    global model
    model = joblib.load('model.pkl')
    app.run(port=9000, debug=True)
