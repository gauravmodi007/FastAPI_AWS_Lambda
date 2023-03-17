# FastAPI_AWS_Lambda
This is working code ML Model As Fast API on AWS Lambada 

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

import uvicorn

import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_text as text

import pickle

import re

import nltk

 

from sklearn.model_selection import train_test_split

from tensorflow import keras

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from pickle import load

 

from nltk.corpus import stopwords

nltk.download('stopwords')

 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_extraction.text import CountVectorizer

 

 

 

 

# Declaring our FastAPI instance

app = FastAPI()

 

# Defining path operation for root endpoint

@app.get("/")

def main():

    return {

        "message": "Welcome to Amgen AI!"

    }

class request_body(BaseModel):

    AutomationId: str

    DosageForm: str

    Product: str

    ProductID: str

    MasterCase: str

    PCM_Subcase: str

    OccurCountry: str

    PPQ: str

    Notes: str

 

@app.post("/AMD")

def AMD(data: request_body):

    encoder = load(open('encoder.pkl', 'rb'))

    amd_model = keras.models.load_model("tf_bert_model")

    clean = re.compile('<.*?>')

    text = re.sub(clean, '', str(data.Notes))

    text = re.sub(r"[^a-zA-Z]"," ",str(text))

    text = " ".join(text.split())

    predicted = amd_model.predict([text])

    l1 = [np.sort(predicted)[0][8],np.sort(predicted)[0][7],np.sort(predicted)[0][6]]

    l2 = encoder.inverse_transform([np.argsort(predicted)[0][8],np.argsort(predicted)[0][7],np.argsort(predicted)[0][6]]).tolist()

    df = pd.DataFrame(list(zip(l2, l1)), columns =['key', 'value'])

    #return {pd.DataFrame(df).to_string()}

    return {

        "AutomationId": data.AutomationId,

        "DosageForm": data.DosageForm,

        "Product": data.Product,

        "ProductID": data.ProductID,

        "MasterCase": data.MasterCase,

        "PCM_Subcase": data.PCM_Subcase,

        "OccurCountry": data.OccurCountry,

        "PCM_ISSUES": [{

        "verbatim": data.Notes,

        "report_codes": [

            {"reported_code": df["key"].iloc[0], "item_type": "Auto-mated Mini Doser", "confidence": df["value"].iloc[0].tolist()},

            {"reported_code": df["key"].iloc[1], "item_type": "Auto-mated Mini Doser", "confidence": df["value"].iloc[1].tolist()},

            {"reported_code": df["key"].iloc[2], "item_type": "Auto-mated Mini Doser", "confidence": df["value"].iloc[2].tolist()},

            ],

        },]

    }

@app.post("/SC")

def SC(data: request_body):

 

    d_arc_Sureclick = {

                        'reported_code': ['Before Activation - resolved', 'autoinjector activation difficulty', 'autoinjector activation failed', 'autoinjector barrel damaged/defective',

                        'autoinjector user mishandling', 'customer feedback','drug appearance', 'drug injection',

                        'drug particles', 'needle bent', 'needle damaged/defective', 'needle shield damaged/defective',

                        'needle shield difficulty', 'safety cover damaged/defective', 'syringe damaged/defective'],

                        'item_type': ['Autoinjector', 'Autoinjector', 'Autoinjector', 'Autoinjector', 'Autoinjector',

                        'Autoinjector', 'Autoinjector', 'Autoinjector', 'Autoinjector', 'Autoinjector',

                        'Autoinjector', 'Autoinjector', 'Autoinjector', 'Autoinjector', 'Autoinjector']  

                        }

 

    reported_codes_arc_Sureclick = pd.DataFrame(data = d_arc_Sureclick)

 

    test_data = [[data.PPQ, data.Notes]]

    input_df = pd.DataFrame(test_data, columns = ['PPQ', 'Notes'])

    model_arc_Sureclick = keras.models.load_model('Sureclick_arc_model')

   

    results_arc = model_arc_Sureclick.predict([input_df.iloc[0][1]])

    results2_arc = results_arc.transpose()

    results_arc_df = pd.DataFrame(results2_arc, columns = ['confidence'])

    results_arc_df.iloc[:,0] *= 100

    results_arc_df['confidence'] = results_arc_df['confidence'].astype(int)

 

    output_arc = pd.concat([reported_codes_arc_Sureclick, results_arc_df], axis = 1)

    output_arc = output_arc.nlargest(3, 'confidence')

    output_arc['confidence'] = output_arc['confidence'].astype(str)

   

    return { #"message": "This is SC API"

            "AutomationId": data.AutomationId,

            "DosageForm": data.DosageForm,

            "Product": data.Product,

            "ProductID": data.ProductID,

            "MasterCase": data.MasterCase,

            "PCM_Subcase": data.PCM_Subcase,

            "OccurCountry": data.OccurCountry,

            "PCM_ISSUES": [{

            "verbatim": data.Notes,

            "report_codes": [

                {"reported_code": output_arc.iloc[0, 0], "item_type": output_arc.iloc[0, 1], "confidence": output_arc.iloc[0, 2]},

                {"reported_code": output_arc.iloc[1, 0], "item_type": output_arc.iloc[1, 1], "confidence": output_arc.iloc[1, 2]},

                {"reported_code": output_arc.iloc[2, 0], "item_type": output_arc.iloc[2, 1], "confidence": output_arc.iloc[2, 2]},

            ],

        },]

    }

 

 

@app.post("/CA")

def CA(data: request_body): return {

        "message": "This is Cartridge API"

    }

@app.post("/DDD")

def DDD(data: request_body):

    file = open('DDD2_model.pk1', 'rb')

    loaded_model = pickle.load(file)

    vectorizer = pickle.load(open("vector.pickel", "rb"))

   

    clean = re.compile('<.*?>')

    text = re.sub(clean, '', str(data.Notes))

    text = re.sub(r"[^a-zA-Z]"," ",str(text))

    text = vectorizer.transform([text])

    text = text.toarray()

    text

   

    predictions = loaded_model.classes_[np.argsort(loaded_model.predict_proba(text))[:, :-3 - 1:-1]].tolist()

    guesses = loaded_model.predict_proba(text)

    probabilities = [sorted(probas, reverse=True)[:3] for probas in guesses]

   

    df1=pd.DataFrame(probabilities).T

    df1.columns=(['value'])

    df3=pd.DataFrame(predictions).T

    df3.columns=(['item_type'])

    df2=df3.replace({'item_type':{'red LED - no audible alarm - after healthcare visit' : 'a0f1N00000FvXDGQA3',

                        'red LED with audible alarm - after healthcare visit' : 'a0f1N00000FvWbFQAV',

                        'adhesive issue at skin' : 'a0f1N00000FvXCXQA3',

                        'patient user error - other' : 'a0f1N00000FvWtNQAV',

                        'adhesive issue at skin with adhesive extender' : 'a0f3l00000ERMQwAAP',

                        'Patient user error- other with adhesive extender' : 'a0f3l00000ERMR1AAP',

                        'HCP device filling error/difficulty' : 'a0f1N00000FvXDuQAN',

                        'HCP device application error/difficulty.' : 'a0f1N00000FvXBwQAN'

                       }})

    df2.columns=(['key'])

    df2

   

    return {

        "AutomationId": data.AutomationId,

        "DosageForm": data.DosageForm,

        "Product": data.Product,

        "ProductID": data.ProductID,

        "MasterCase": data.MasterCase,

        "PCM_Subcase": data.PCM_Subcase,

        "OccurCountry": data.OccurCountry,

        "PCM_ISSUES": [{

            "verbatim": data.Notes,

            "report_codes": [

                {"reported_code": df2["key"].iloc[0], "item_type": df3["item_type"].iloc[0], "confidence": df1["value"].iloc[0].tolist()},

                {"reported_code": df2["key"].iloc[1], "item_type": df3["item_type"].iloc[1], "confidence": df1["value"].iloc[1].tolist()},

                {"reported_code": df2["key"].iloc[2], "item_type": df3["item_type"].iloc[2], "confidence": df1["value"].iloc[2].tolist()},

                ],

            },]

    }

   

    

@app.post("/PFS")

def PFS(data: request_body): return {

        "message": "This is pre-filled syringe API"

    }
