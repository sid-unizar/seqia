
from . text_cleaning import clean_text
import json
from tqdm import tqdm
import tarfile
import os
import tempfile

#Mapping between expected JSON field values and the ones in custom files
#(Can be overridden via an external file)
JSON_mapping = {
    'body': 'articleBody',
    'headline': 'headline'
}

#Function to load an article from a JSON file (uses custom mapping of JSON fields if defined)
def load_article_from_json_file(f,filename):
    
    article = {}

    article['filename'] = filename
    article['drought'] = False
    article['impacts'] = []

    try:
        art_json = json.load(f)

        try:
            article['headline'] = clean_text(art_json[JSON_mapping['headline']])
        except:
            article['headline'] = ''

        try:
            article['body'] = clean_text(art_json[JSON_mapping['body']])
        except:
            article['body'] = ''

        article['loaded'] = True
    except:
        article['headline'] = ''
        article['body'] = ''
        article['loaded'] = False
    
    return article

#Functions to load articles from either a folder or a TAR file
def load_articles_from_folder(path):

    articles = []

    for dirpath,_,files in os.walk(path):
        for file in tqdm(files,desc='Loading articles from folder'):
            if file.endswith('.json'):
                #with open(os.sep.join([dirpath, file]),encoding= 'utf-8') as f:
                with open(os.sep.join([dirpath, file]),'rb') as f:
                    #Workaround to fix encoding issues: we read each file as a binary
                    #file, then decode it to UTF-8. This won't affect well-formatted
                    #Unicode files, but it will convert to a desired format an ISO-encoded
                    #file. Each time we read this file, we create a temporary file, which
                    #will serve as the input to the reading function below. This way, we
                    #can cleanly keep the existing code without many modifications
                    tp = tempfile.TemporaryFile(mode='r+',encoding='utf-8')
                    tp.write(f.read().decode('utf-8','ignore'))
                    tp.seek(0)
                    articles.append(load_article_from_json_file(tp,file))
                    tp.close()

    return articles

def load_articles_from_tar(tar_filename):

    articles = []

    tar = tarfile.open(tar_filename)

    for file in tqdm(tar.getmembers(),desc='Loading articles from TAR file'):
        if file.name.endswith('.json'):
            tp = tempfile.TemporaryFile(mode='r+',encoding='utf-8')
            tp.write(tar.extractfile(file.name).read().decode('utf-8','ignore'))
            tp.seek(0)
            articles.append(load_article_from_json_file(tp,file.name))
            tp.close()
    
    return articles

#Function for loading mapping for custom article JSON files (if defined)
def load_custom_json_mapping(mapping_file):
    mapping = dict()
    with open(mapping_file,'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            line = line.split('\t')
        mapping[line[0]] = line[1]
    return mapping
