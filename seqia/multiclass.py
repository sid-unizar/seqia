
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import numpy as np
from . dataset import DroughtDataset

class MulticlassClassifier:
    
    tokenizer = None
    model = None
    trainer = None

    MULTICLASS_MODEL_MAX_SIZE = 512
    multiclass_base_model_name = "PlanTL-GOB-ES/roberta-base-bne"

    id2label = {
        1: 'Agriculture',
        2: 'Farming',
        3: 'Hydrology',
        4: 'OTHERS'
    }
    
    #Constructor
    def __init__(self,modelPath=''):
        self.tokenizer = self.load_multiclass_tokenizer()
        self.model = self.load_multiclass_classifier(modelPath)

        training_args = TrainingArguments(output_dir='/',auto_find_batch_size = True, disable_tqdm=True)
        
        self.trainer = Trainer(
                model=self.model,
                args=training_args,
        )

        return
    
    #Functions to load the multiclass model and tokenizer
    def load_multiclass_classifier(self,modelPath=''):
        multiclass = None
        if modelPath != '':
            if os.path.isfile(modelPath + 'pytorch_model.bin'):
                multiclass = AutoModelForSequenceClassification.from_pretrained(modelPath, num_labels=len(self.id2label.keys()))
        else:
            multiclass = AutoModelForSequenceClassification.from_pretrained(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')),'multiclass'), num_labels=len(self.id2label.keys()))

        return multiclass

    def load_multiclass_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.multiclass_base_model_name)
    
    #Inference call function
    def __call__(self, texts : list):
        labels_sentences = set()
        dataset = DroughtDataset(self.tokenizer(texts, max_length=self.MULTICLASS_MODEL_MAX_SIZE, pad_to_max_length=True,truncation=True), len(texts))

        predictions,_,_ = self.trainer.predict(dataset)
            
        probs = list(torch.argmax(torch.Tensor(predictions), axis=-1))
            
        for prob in probs:
            labels_sentences.add(self.id2label[int(prob)+1])

        if 'OTHERS' in labels_sentences:
            labels_sentences.discard('OTHERS')
        
        return list(labels_sentences)
