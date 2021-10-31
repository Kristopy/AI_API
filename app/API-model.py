import pathlib

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

class AI_MODELS():

    #TODO you should implement numerous  functions implementing different AI-MODELS. 
    #TODO Be sure to implement the models in a neat fashion in order to remain in control when multiple models are in place
    
    def __init__(self):
        #? What should be added here?
        self.BASE_DIR = pathlib.Path().resolve(__file__)
        self.DATASETS_DIR = BASE_DIR / 'Datasets'

    #* AI for Determining spam or ham - based on datasets_SMS
    def AI_SPAM_HAM():
        DATASETS_SMS = DATASETS_DIR / 'Datasets_SMS'
        EXPORT_DIR = DATASETS_SMS / 'Exports'
        #Three main targets to unpack
        MODEL_EXPORT_PATH = EXPORT_DIR / 'Spam_Model.h5'
        METADATA_EXPORT_PATH = EXPORT_DIR / 'Spam-Metadata.json'
        TOKENIZER_EXPORT_PATH = EXPORT_DIR / 'Spam-Tokenizer.json'
        print(MODEL_EXPORT_PATH)

    #* AI for Number recognition - based on datasets_Num_Rec
    def AI_NUM_REC():
        DATASETS_NUM_REC = DATASETS_DIR / 'Datasets_NUM_REC'
        EXPORT_DIR = DATASETS_NUM_REC / 'Exports'
        MODEL_EXPORT_PATH = EXPORT_DIR / 'Num_Rec_Model.h5'
        
        
        print(MODEL_EXPORT_PATH)

