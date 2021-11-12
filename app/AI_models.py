import json
import numpy as np

from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass  # pip install dataclasses

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from . import Images_address

@dataclass
class AIModel:

    # * Generally declearing __init__ variabled -> for use in all models implemented
    model_path: Path
    tokenizer_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    image_path: Optional[Path] = None

    model = None
    tokenizer = None
    metadata = None
    image = None
    # * __post_inti__ for checking if model and other important files are being implemented
    def __post_init__(self):
        if self.model_path.exists():
            self.model = load_model(self.model_path)
        if self.tokenizer_path:
            if self.tokenizer_path.exists():
                if self.tokenizer_path.name.endswith("json"):
                    tokenizer_text = self.tokenizer_path.read_text()
                    self.tokenizer = tokenizer_from_json(tokenizer_text)
        if self.metadata_path:
            if self.metadata_path.exists():
                if self.metadata_path.name.endswith("json"):
                    self.metadata = json.loads(self.metadata_path.read_text())
        if self.image_path:
            if self.image_path.exists():
                #Convert image if image exist - take in path
                self.image = Images_address.Image_Convert(self.image_path)


    # * get function for fetching models and other important elements
    def get_model(self):
        if not self.model:
            raise Exception("Model not implemeted")
        return self.model
   
    def get_tokenizer(self):
        if not self.tokenizer:
            raise Exception("tokenizer not implemeted")
        return self.tokenizer
    
    def get_metadata(self):
        if not self.metadata:
            raise Exception("metadata not implemeted")
        return self.metadata
    
    def get_image(self):
        if not self.image:
            raise Exception("Image not found for converting")
        return self.image


    # * This section is based upon SMS-model. 
    # * START
    def label_legend_inverted(self):
        label_legend_inverted = self.get_metadata()['label_legend_inverted']
        return label_legend_inverted

    def Results_preds(self, query: str):
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        metadata = self.get_metadata()    

        # Converting input text to sequences from tokenizer
        sequences = tokenizer.texts_to_sequences([query])
        maxlen = metadata.get('max_seq_len') or 280

        # Padding the x-input for formatting
        x_input = pad_sequences(sequences, maxlen=maxlen)

        # passing in x-input in correct format and sequence to model.predict()
        y_output = model.predict(x_input)
        
        #top_y_input = np.argmax(y_output) #Collecting index of largest value example: ([0.9837, 0.0167]), yields index 0
        preds = y_output[0]
        top_idx_val = np.argmax(preds)

        top_pred = {'labels': self.label_legend_inverted()[str(top_idx_val)],
                    'Confidence': float(preds[top_idx_val])}

        labeled_preds = [{'labels': self.label_legend_inverted()[str(i)],
                        'Confidence': float(x)} for i, x in enumerate(preds)]

        return {'top': top_pred, 'Predictions': labeled_preds}
    # * END

    # * This section is based upon AI_NUM_REC MODEL.
    # * START
    # def Convert_img_to_vec(self, query: str):
    #     try: 
    #         vec_image = Image_Convert()
    #         return vec_image
    #     except:
    #         return Exception('File not read!')
        
        

    def AI_NUM_REC(self, path_to_img: str):
        model = self.get_model()
        metadata = self.get_metadata()
        image = self.get_image() #Should be a matrice list 

        val_acc = metadata['val_acc']
        val_loss = metadata['val_loss']
        # X_test = metadata['X_test'] #* When dumped to json we converted the matrix by applying .to_list()
       
        #! X_test consist of multiple matrices of numbers between 0-9
        # THis vec_image should contain only one image array not multiple.
        predictions = model.predict([image])
        #! Should return what we want to see in API - input query should be something clever - working on in
        #TODO Query should be something reasenable possibly the path to image. 
        if not path_to_img.exists():
            return {'Model Criteria': 'Model is based upon images from 0-9, inputs above this is value not valid',
                    'Exception Error': 'Path not found for image - pass in your local path'}
        else:
            return {'Validity accuracy': val_acc,
                    'Validity loss': val_loss,
                    'Predictions': np.argmax(predictions),
                    'Correct Output': str(path_to_img)}
     # * END

