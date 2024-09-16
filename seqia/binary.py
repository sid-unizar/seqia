
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
from . dataset import DroughtDataset

class BinaryClassifier:

    BINARY_MODEL_MAX_SIZE = 4096
    binary_base_model_name = 'PlanTL-GOB-ES/longformer-base-4096-bne-es'

    #Constructor
    def __init__(self,modelPath=''):
        self.tokenizer = self.load_binary_tokenizer()
        self.model = self.load_binary_classifier(modelPath)

        training_args_binary = TrainingArguments(output_dir='/',auto_find_batch_size = True)
        self.trainer = Trainer(
            model=self.model,
            args=training_args_binary,
        )

        return
    
    #Functions to load the binary model and tokenizer
    def load_binary_classifier(self,modelPath=''):
        binary = None
        if modelPath != '':
            if os.path.isfile(modelPath + 'pytorch_model.bin'):
                binary = AutoModelForSequenceClassification.from_pretrained(modelPath, num_labels=2)
        else:
            binary = AutoModelForSequenceClassification.from_pretrained(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')),'binary_model'), num_labels=2)

        return binary

    def load_binary_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.binary_base_model_name)
    
    #Inference call function
    def __call__(self, articles, include_or_not_list):

        results = []

        encodings = self.tokenizer([str(article['headline'] + '~' + article['body']) for i, article in enumerate(articles) if include_or_not_list[i] == 1],max_length=self.BINARY_MODEL_MAX_SIZE, pad_to_max_length=True,truncation=True)
        num_of_articles = len([i for i in range(len(articles)) if include_or_not_list[i] == 1])
        dataset = DroughtDataset(encodings, num_of_articles)
        
        logits_binary,_,_ = self.trainer.predict(dataset)
        predictions_binary = list(np.argmax(logits_binary, axis=-1))
    	
        next_positive_index = 0
        for _, prev_result in include_or_not_list.items():
            if prev_result == 0:
                results.append(0)
            else:
                results.append(int(predictions_binary[next_positive_index]))
                next_positive_index += 1

        return results
