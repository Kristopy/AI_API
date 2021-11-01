import json
import numpy as np

from typing import Optional, List
from pathlib import Path
import pathlib
from dataclasses import dataclass  # pip install dataclasses

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


@dataclass
class AIModel:
    model_path: Path
    tokenizer_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    model = None
    tokenizer = None
    metadata = None

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

    
    def example(self):
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        metadata = self.get_metadata()
        print(model, '\n')
        print(tokenizer, '\n')
        print(metadata, '\n')



