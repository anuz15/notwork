
from flask import Flask,request,jsonify,url_for,render_template

import numpy as np
import pickle as p
import pandas as pd
app = Flask(__name__)
rfmodel = p.load(open('rfmodel','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    data=np.array(list(data.values())).reshape(1,-1)
    output=rfmodel.predict(data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "_main_":
    app.run(debug=True)