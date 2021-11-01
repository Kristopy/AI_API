import numpy as np
import json
import pathlib

from typing import Optional
from fastapi import FastAPI

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


app = FastAPI()

BASE_DIR = pathlib.Path().resolve(__file__)
DATASETS_DIR = BASE_DIR / 'Datasets'
EXPORT_DIR = DATASETS_DIR / 'Exports'

#Three main targets to unpack
MODEL_EXPORT_PATH = EXPORT_DIR / 'Spam_Model.h5'
METADATA_EXPORT_PATH = EXPORT_DIR / 'Spam-Metadata.json' 
TOKENIZER_EXPORT_PATH = EXPORT_DIR / 'Spam-Tokenizer.json'

AI_MODEL = None
AI_TOKENIZER = None
MODEL_METADATA = {}
label_legend_inverted = {}


@app.on_event('startup')
def on_startup():
    global AI_MODEL, AI_TOKENIZER, MODEL_METADATA, label_legend_inverted

    #Load Models: 
    if MODEL_EXPORT_PATH.exists():
        AI_MODEL = load_model(MODEL_EXPORT_PATH)
   
    if TOKENIZER_EXPORT_PATH.exists():
        t_json = TOKENIZER_EXPORT_PATH.read_text()
        AI_TOKENIZER = tokenizer_from_json(t_json)

    if METADATA_EXPORT_PATH.exists():
        MODEL_METADATA = json.loads(METADATA_EXPORT_PATH.read_text())
        label_legend_inverted = MODEL_METADATA['label_legend_inverted']
        #label_legend_inverted['0'] = 'valid'

def predict(query:str):
    global AI_MODEL, AI_TOKENIZER, MODEL_METADATA, label_legend_inverted

    # Converting input text to sequences from tokenizer
    sequences = AI_TOKENIZER.texts_to_sequences([query])
    maxlen = MODEL_METADATA.get('max_seq_len') or 280
    # Padding the x-input for formatting
    x_input = pad_sequences(sequences, maxlen=maxlen)
    # passing in x-input in correct format and sequence to model.predict()
    y_output = AI_MODEL.predict(x_input)
    #top_y_input = np.argmax(y_output) #Collecting index of largest value example: ([0.9837, 0.0167]), yields index 0
    preds = y_output[0]

    top_idx_val = np.argmax(preds)

    top_pred = {'labels': label_legend_inverted[str(top_idx_val)],
                'Confidence': float(preds[top_idx_val])}

    labeled_preds = [{'labels': label_legend_inverted[str(i)], 
                      'Confidence': float(x)} for i, x in enumerate(preds)]
    
    return {'top':top_pred, 'Predictions':labeled_preds}

@app.get("/")
async def read_index(q: Optional[str] = None):  # /?q=Something he
    global AI_MODEL, AI_TOKENIZER, MODEL_METADATA, label_legend_inverted

    query = q or 'Hello world'
    preds_dict = predict(query)
    return {'AI-Model': 'AI_MODEL',
            'Query': query,
            'Results': preds_dict
            }