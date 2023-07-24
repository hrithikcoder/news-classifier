from flask import Flask,request,jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

model=pickle.load(open("fakenews_model.pkl",'rb'))
vector=pickle.load(open("vectorizer.pkl",'rb'))





app = Flask(__name__)

@app.route('/')
def home():
    return "hello World"

@app.route('/predict',methods=['POST'])
def predict():
    news=str(request.form['news'])
    result=model.predict(vector.transform([news]))[0]
    return jsonify(result)

if __name__ == '__main__':

    app.run()
