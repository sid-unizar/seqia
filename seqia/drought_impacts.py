
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import numpy as np
from . dataset import DroughtDataset
from tqdm import tqdm

class DroughtImpactsClassifier:

    impacts_and_base_model = {
        'Agricultura': ("PlanTL-GOB-ES/roberta-base-bne", 512),
        'Ganadería': ("PlanTL-GOB-ES/roberta-base-bne", 512),
        'Recursos_hídricos': ("PlanTL-GOB-ES/roberta-base-bne", 512),
        'Energético': ("PlanTL-GOB-ES/roberta-base-bne", 512)
    }

    
    #Constructor
    def __init__(self,modelPath=''):

        self.usesSingleTokenizer = True
        self.tokenizer = dict()
        self.model = dict()
        self.trainer = dict()

        #We define a support function that loads a custom text file
        #in the repository that might contain a different list of
        #drought impacts than those that are hard-coded above. The
        #function accepts one optional parameter, which is a boolean
        #that tells the function whether we want to load the drought
        #impacts from an external folder or not. If it is set to False,
        #it will not load anything and use the drought impacts defined
        #in the class variables above; if True is declared, then it will
        #read a text file called "impacts.txt", from the repository folder,
        #in which it will list all supported drought impacts by name, as 
        #well as the URL to their original pre-trained models for each one
        #of them
        self.impacts_and_base_model = self.load_drought_impacts_from_file()

        model_names_unique = set()
        for _, model_base_names in self.impacts_and_base_model.items():
            model_names_unique.add(model_base_names[0])
        if len(list(model_names_unique)) > 1:
            self.usesSingleTokenizer = False

        training_args = dict()

        for impact in self.impacts_and_base_model.keys():
            #Load all individual binary classifiers for each of the supported drought impacts
            self.tokenizer[impact] = self.load_drought_impacts_tokenizer(impact=impact)
            self.model[impact] = self.load_individual_drought_impacts_classifier(impact=impact, modelPath=modelPath)

            training_args[impact] = TrainingArguments(output_dir='/',auto_find_batch_size = True, disable_tqdm=True)
            
            self.trainer[impact] = Trainer(
                    model=self.model[impact],
                    args=training_args[impact],
            )

        return
    
    def load_drought_impacts_from_file(self,loadFromExternalFile=False):

        ret = {}

        if loadFromExternalFile:
            if os.path.join(os.path.dirname(os.path.realpath(__file__)), 'impacts.txt'):
                with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'impacts.txt')) as f:
                    for line in f:
                        if line[-1] == '\n':
                            line = line[:-1]
                        line = line.split('\t')
                        ret[line[0]] = line[1]
        else:
            ret = self.impacts_and_base_model
        return ret

    def load_drought_impacts_tokenizer(self,impact):
        return AutoTokenizer.from_pretrained(self.impacts_and_base_model[impact][0])
    
    #Functions to load each individual model for each drought impact
    def load_individual_drought_impacts_classifier(self,impact,modelPath=''):
        model = None
        if modelPath != '':
            if os.path.isfile(modelPath + 'pytorch_model.bin'):
                model = AutoModelForSequenceClassification.from_pretrained(modelPath, num_labels=2)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')),'impacts'), impact), num_labels=2)
        return model

    #Inference call function
    def __call__(self, texts):

        results = dict()
        
        #Pass the sentences through each of the individual classifiers
        for positive_idx, text, _, _ in tqdm(texts):
            result_cur = list()
            for impact in self.impacts_and_base_model.keys():
                if self.usesSingleTokenizer:
                    logits_binary = self.model[impact](**self.tokenizer[list(self.impacts_and_base_model.keys())[0]](text, max_length=self.impacts_and_base_model[list(self.impacts_and_base_model.keys())[0]][1], pad_to_max_length=True,truncation=True,return_tensors='pt'))
                else:
                    logits_binary = self.model[impact](**self.tokenizer[impact](text, max_length=self.impacts_and_base_model[impact][1], pad_to_max_length=True,truncation=True,return_tensors='pt'))
                predictions_binary = list(np.argmax(logits_binary.logits.detach().numpy(), axis=-1))

                if 1 in predictions_binary:
                    result_cur.append(impact)
            
            results[positive_idx] = result_cur

        return results
    
