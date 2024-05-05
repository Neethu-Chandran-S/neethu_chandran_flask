from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


app=Flask(__name__)
iris=load_iris()
X,y=iris.data,iris.target

clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X,y)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def predict():
 
  features=[float(request.form['SL']),
           float(request.form['SW']),
           float(request.form['PL']),
           float(request.form['PW'])]
 

   
   
  prediction = clf.predict([features])[0]
  flower=iris.target_names[prediction]
   


  return render_template('prediction.html',flower=flower)

if __name__ == '__main__':
     app.run()