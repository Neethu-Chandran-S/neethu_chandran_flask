from flask import Flask,render_template,request
import numpy as np
app=Flask(__name__)
import pickle

model=pickle.load(open('model_new.pkl','rb'))


@app.route('/')
def hello():
    return render_template('index.html')
@app.route('/prediction',methods=['GET','POST'])
def predict():
   sl=float(request.form['SL'])
   sw=float(request.form['SW'])
   pl=float(request.form['PL'])
   pw=float(request.form['PW'])

   input = np.array([sl,sw,pl,pw])
   input = input.reshape(1,-1)
   prediction = model.predict(input)
   prediction = prediction.item()

   return render_template('prediction.html',flower="The predicted species is '{}'.".format(prediction))


if __name__ == '__main__':
     app.run()