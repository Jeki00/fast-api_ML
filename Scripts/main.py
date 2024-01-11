from fastapi import FastAPI, Body
import joblib
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


app = FastAPI()

def cleaningUlasan(text):
    text = re.sub(r'[0-9]+',' ',text)
    text = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", text)
    text = text.strip(' ')
    return text

def caseFolding(text):
    text = text.lower()
    return text

def tokenizingText(text):
    text = word_tokenize(text)
    return text

stop_words = set(stopwords.words('indonesian'))
def stopwordRemoval(text):
    return [word for word in text if word not in stop_words]

def joinWord(text):
    return " ".join(text)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(text):
    text = [stemmer.stem(word) for word in text]
    return text

def convertToLabel(input):
    if input == 0:
        return "negative"
    elif input == 2:
        return "positive"
    else:
        return "neutral"



def preprocessing(input):
    input = tokenizingText(input)
    input = stopwordRemoval(input)
    input = stemming(input)
    input = joinWord(input)

    return [input]

vectorizer = joblib.load('model/vectorizer.joblib')

@app.post("/svm", status_code=201)
async def read_root(input: str = Body(...)):
    inp = preprocessing(input)
    inp = vectorizer.transform(inp)
    SVM = joblib.load('model/SVM.joblib')

    return {"ulasan":input,"result":convertToLabel(SVM.predict(inp))}

@app.post("/knn", status_code=201)
async def read_root(input: str = Body(...)):
    
    inp = preprocessing(input)
    inp = vectorizer.transform(inp)
    KNN = joblib.load('model/KNN.joblib')

    return {"ulasan":input,"result":convertToLabel(KNN.predict(inp))}

@app.post("/lr", status_code=201)
async def read_root(input: str = Body(...)):
    inp = preprocessing(input)
    inp = vectorizer.transform(inp)
    LR = joblib.load('model/LR.joblib')

    return {"ulasan":input,"result":convertToLabel(LR.predict(inp))}
