#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
def model_creation():
    df=pd.read_csv("airfoil_self_noise.dat",sep="\t",header=None)
    print(df)
    df.columns=["Frequency","Angle of attack","Chord length","Free-stream velocity","Suction side displacement","Scaled sound pressure"]
    df.isnull().sum()
    X=df.iloc[:,:-1]
    Y=df.iloc[:,-1]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)
    sns.pairplot(X_train)
    plt.show()
    regressor=LinearRegression()
    regressor.fit(X_train,Y_train)
    pickle.dump(regressor,open("model.pkl","wb"))
# model_creation()
    

    
import flask
from flask import Flask,request,app,jsonify,url_for,render_template
from flask import Response
from flask_cors import CORS 
app=Flask(__name__,template_folder='template')
model=pickle.load(open("model.pkl","rb"))

#This is code to test it by using postman
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     new_data=[list(data.values())]
#     output=model.predict(new_data)[0]
#     return jsonify(output)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    find_features=[np.array(data)]
    print(data)
    output=model.predict(find_features)[0]
    return render_template('home.html',prediction_text="Airfoil Pressure is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)