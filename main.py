from flask import Flask, request, render_template
import pickle as pk
import numpy as np

# Flask constructor
main = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@main.route('/')
def xyz():
    return render_template('index.html')

# A decorator used to tell the application
# which URL is associated function
# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,10)
    loaded_model = pk.load(open("stroke_finalized_model.pk", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@main.route('/result', methods=['POST'])
def gfg():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Chance of Stroke'
        else:
            prediction = 'No Chance of Stroke'
        return render_template("result.html", prediction=prediction)


if __name__ == '__main__':
    main.run(debug=True)
