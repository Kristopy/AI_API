import pathlib

from typing import Optional
from fastapi import FastAPI
from . import AI_models

app = FastAPI()

#TEST

BASE_DIR = pathlib.Path().resolve(__file__)
DATASETS_DIR = BASE_DIR / 'Datasets'

DATASETS_SMS_DIR = DATASETS_DIR / 'Datasets_SMS'
DATASETS_NUM_REC_DIR = DATASETS_DIR / 'Datasets_NUM_REC'

EXPORT_SMS_DIR = DATASETS_SMS_DIR / 'Exports'
EXPORT_NUM_REC_DIR = DATASETS_NUM_REC_DIR / 'Exports'

#Three main targets to unpack
MODEL_SMS_EXPORT_PATH = EXPORT_SMS_DIR / 'Spam_Model.h5'
TOKENIZER_EXPORT_PATH = EXPORT_SMS_DIR / 'Spam-Tokenizer.json'
METADATA_EXPORT_PATH = EXPORT_SMS_DIR / 'Spam-Metadata.json'

MODEL_NUM_REC_EXPORT_PATH = EXPORT_NUM_REC_DIR / 'Num_Rec_Model.h5'
METADATA_NUM_REC_EXPORT_PATH = EXPORT_NUM_REC_DIR / 'Num_rec_Metadata.json'

IMAGE_DIR = DATASETS_NUM_REC_DIR / 'Images_convert'
IMAGE_PATH = IMAGE_DIR / 'Number_0.png'

@app.get("/SMS")
async def read_index(q: Optional[str] = None):  # /?q=Something he
    global AI_MODEL

    AI_MODEL = AI_models.AIModel(
        model_path=MODEL_SMS_EXPORT_PATH,
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
async def read_index(q: Optional[str] = None):  # /?q=path to filename
    global AI_MODEL

    #Seems like the or method did not work with pathlib.Path module implemented
    if q == None:
        query = IMAGE_PATH #default path if q == None
    else:
        query = pathlib.Path(q)

    AI_MODEL = AI_models.AIModel(
        model_path=MODEL_NUM_REC_EXPORT_PATH,
        metadata_path=METADATA_NUM_REC_EXPORT_PATH,
        image_path=query
    )
    
    results = AI_MODEL.AI_NUM_REC(query)


    return {'AI-Model': 'Number recognition from images',
            'Query': query,
            'Results': results
            }
