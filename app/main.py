import numpy as np
import json
import pathlib

from typing import Optional
from fastapi import FastAPI

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from . import AI_models

app = FastAPI()

BASE_DIR = pathlib.Path().resolve(__file__)
DATASETS_DIR = BASE_DIR / 'Datasets'

DATASETS_SMS_DIR = DATASETS_DIR / 'Datasets_SMS'
DATASETS_NUM_REC_DIR = DATASETS_DIR / 'Datasets_NUM_REC'

EXPORT_SMS_DIR = DATASETS_SMS_DIR / 'Exports'
EXPORT_NUM_REC_DIR = DATASETS_NUM_REC_DIR / 'Exports'

#Three main targets to unpack
MODE_SMS_EXPORT_PATH = EXPORT_SMS_DIR / 'Spam_Model.h5'
TOKENIZER_EXPORT_PATH = EXPORT_SMS_DIR / 'Spam-Tokenizer.json'
METADATA_EXPORT_PATH = EXPORT_SMS_DIR / 'Spam-Metadata.json'

MODEL_NUM_REC_EXPORT_PATH = EXPORT_NUM_REC_DIR / 'Num_Rec_Model.h5'
METADATA_NUM_REC_EXPORT_PATH = EXPORT_NUM_REC_DIR / 'Num_rec_Metadata.json'

# @app.on_event('startup')
# def on_startup():
#     global AI_MODEL, AI_TOKENIZER, MODEL_METADATA, label_legend_inverted

#     #Load Models: 
#     if MODEL_EXPORT_PATH.exists():
#         AI_MODEL = load_model(MODEL_EXPORT_PATH)
   
#     if TOKENIZER_EXPORT_PATH.exists():
#         t_json = TOKENIZER_EXPORT_PATH.read_text()
#         AI_TOKENIZER = tokenizer_from_json(t_json)

#     if METADATA_EXPORT_PATH.exists():
#         MODEL_METADATA = json.loads(METADATA_EXPORT_PATH.read_text())
#         label_legend_inverted = MODEL_METADATA['label_legend_inverted']
#         #label_legend_inverted['0'] = 'valid'
#HELlo 

@app.get("/SMS")
async def read_index(q: Optional[str] = None):  # /?q=Something he
    global AI_MODEL

    AI_MODEL = AI_models.AIModel(
        model_path=MODE_SMS_EXPORT_PATH,
        tokenizer_path=TOKENIZER_EXPORT_PATH,
        metadata_path=METADATA_EXPORT_PATH
    )

    query = q or 'Hello world'
    preds_dict = AI_MODEL.Results_preds(query)

    return {'AI-Model': 'SMS - SPAM and HAM estimation',
            'Query': query,
            'Results': preds_dict
            }


@app.get("/NUM_REC")
async def read_index(q: Optional[int] = None):  # /?q=Something he
    global AI_MODEL

    AI_MODEL = AI_models.AIModel(
        model_path=MODEL_NUM_REC_EXPORT_PATH,
        metadata_path=METADATA_NUM_REC_EXPORT_PATH
    )
    
    query = q or 1
    return {'AI-Model': 'Number recognition from images',
            'Query': query,
            'Results': AI_MODEL.AI_NUM_REC(query)
            }
