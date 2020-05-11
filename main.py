from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
def data_split(data, ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)                               
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
# data Read
df=pd.read_csv("data.csv")
train, test=data_split(df, 0.2)
x_train=train[["Fever", "Body Pain", "Age", "Runny Nose", "Diff Breathe"]].to_numpy()
x_test=test[["Fever", "Body Pain", "Age", "Runny Nose", "Diff Breathe"]].to_numpy()
y_train=train[["Infection"]].to_numpy().reshape(1980 ,)
y_test=test[["Infection"]].to_numpy().reshape(494 ,)
clf=LogisticRegression()
clf.fit(x_train, y_train)
# OPen a file
file=open("model.pkl", "wb")
pickle.dump(clf, file)
file.close()


app=Flask(__name__)
file=open("model.pkl", "rb")
clf=pickle.load(file)
file.close()

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        abcd=request.form
        fever=int(abcd['fever'])
        age=int(abcd['age'])
        runnynose=int(abcd['runnyNose'])
        diffbreathe=int(abcd['diffBreathe'])
        pain=int(abcd['pain'])
        inputFeatures=[fever, pain, age, runnynose, diffbreathe]
        Infection=clf.predict_proba([inputFeatures])[0][1]
        
        return render_template("show.html", result=round(Infection*100))
    return render_template("index.html")
if __name__=="__main__":

    app.run(debug=True)
    