from flask import Flask,render_template,request
import numpy as np
import pickle
import math
app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        years = [x for x in request.form.get("years")]   
 
        preds=[np.array(years,dtype='float32')]
        prediction=model.predict(preds)
        p=round(round(prediction[0],2)*12/100000,2)
        
        return render_template('index.html',prediction_text=p )


#@app.route("/sub",methods=['POST'])
#def submit():
   

if __name__=="__main__":
    app.run(debug=True)