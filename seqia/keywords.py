
import re
from tqdm import tqdm

class KeywordClassifier:

    drought_regex = r'(sequia|sequía)+'
    
    def __init__(self):
        return
    
    def __call__(self, articles):
        results = {}
        for i, article in tqdm(enumerate(articles)):
            #headline = article['headline']
            #body = article['body']
            body = article['headline'].lower() + '\t' + article['body'].lower()
            if re.search(self.drought_regex, body):
                results[i] = 1
            else:
                results[i] = 0
        return results
