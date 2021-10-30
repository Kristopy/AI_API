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

print(BASE_DIR, 'Hello world')
