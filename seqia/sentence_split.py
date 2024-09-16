
import spacy
import json

class SentenceSplitter:
    def __init__(self):
        self.nlp = None
        #Load spaCy library for separating texts into sentences
        self.nlp = spacy.load("es_core_news_sm")

    def __call__(self, article):
        try:
            text = article['headline'] + '. ' + article['body']
        except:
            text = article['body']
        doc = self.nlp(text)
        return [str(sentence) for sentence in doc.sents], doc, [sentence.as_doc() for sentence in doc.sents]
        
