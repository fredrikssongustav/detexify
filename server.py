import json

import flask
from flask_cors import CORS

from model.Image.Image import Image
from model.createModel import create_model

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

app = flask.Flask(__name__)
cors = CORS(app)

global model
model = create_model(32, 32, 1098)

model.load_weights("static/weights")

global labelsByIndex

with open('static/labelsByIndex.json') as json_file:
    labelsByIndex = json.load(json_file)

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": True}
    bestGuesses = []

    if flask.request.method == "POST":
        if flask.request.json and 'coordinates' in flask.request.json:
            coordinates = flask.request.json['coordinates']

            image = Image(coordinates, 450, 32)
            preds = model.predict(image.return_as_np(), verbose=0)[0]

            for i in preds.argsort()[-10:][::-1]:
                index = [symbol for symbol in labelsByIndex['symbols'] if symbol['index'] == i][0]
                bestGuesses.append(index)

    data['predictions'] = bestGuesses

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    app.run()
