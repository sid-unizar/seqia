

#Import main libraries
import os
import torch
from collections import defaultdict
from tqdm import tqdm
import warnings

#Models classes
from . binary import BinaryClassifier
from . keywords import KeywordClassifier
#from . multiclass import MulticlassClassifier
from . drought_impacts import DroughtImpactsClassifier
from . ner_loc import NERLocation

#Support functions
from . article_load import load_articles_from_folder
from . dataset import DroughtDataset
from . sentence_split import SentenceSplitter

device = None

#Main class definition
class DroughtClassifier:
    binary = None
    multiclass = None
    ner_location = None
    geonames_username = None
    def __init__(self,gpu=0,cpu_threads=0):

        self.exclude_problematic_articles = False
        self.problematic_articles = []

        #Check if GPU is available for inference mode, else use CPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu":
            warnings.warn("Running inference in CPU. You should better use a GPU.")
            if cpu_threads > 0:
                if cpu_threads > 1:
                    print("Running on CPU with",cpu_threads,"threads. This setting can be changed by calling this class\' \"change_number_cpu_threads()\" method.")
                else:
                    print("Running on CPU with",cpu_threads,"thread. This setting can be changed by calling this class\' \"change_number_cpu_threads()\" method.")
                
                self.change_number_cpu_threads(cpu_threads)
        else:
            torch.cuda.set_device(gpu)  #Outdated function!!!!

        #Load models
        self.binary = BinaryClassifier()

        self.keyword = KeywordClassifier()

        #self.multiclass = MulticlassClassifier()
        self.drought_impacts = DroughtImpactsClassifier()

        self.ner_location = NERLocation()
        
        self.sentence_split = SentenceSplitter()

        #TODO

        return
    
    def change_number_cpu_threads(self,num):
        torch.set_num_threads(num)
        return
        
    def binary_classifier(self, texts : list):
        return self.binary(texts)
    
    #def multiclass_classifier(self, texts: list):
    #    return self.multiclass(texts)

    def detect_repeated_articles(self,articles):
        #First find repeated article entries via a simple checking through
        #Python's sets: if an article has the absolute same body as another
        #loaded news article, then we exclude it
        repeated = []
        texts = set()
        entries = dict()
        for article in tqdm(articles, desc='Checking for duplicate articles'):
            if article['body'] not in texts:
                texts.add(article['body'])
                entries[article['body']] = article
            else:
                repeated.append((article['filename'],'REPEATED_ARTICLE_BODY: ' + entries[article['body']]['filename']))

        #TODO: Implement text-reuse Python bindings to allow for more sophisticated search of duplicates

        return repeated

    def detect_problems_with_articles(self,articles):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/longformer-base-4096-bne-es')
        problems = []
        for article in tqdm(articles, desc='Checking for problems in corpus'):
            #Is the body of the article empty or too short (one to three characters)?
            if len(article['body']) <= 3:
                problems.append((article['filename'],'BODY_TOO_SHORT: ' + str(len(article['body']))))
            
            #Is the headline of the article empty?
            if len(article['headline']) == 0:
                problems.append((article['filename'],'HEADLINE_TOO_SHORT: ' + str(len(article['headline']))))

            #Is the article too long? (more than 4,096 BPE tokens)
            article_length = len(tokenizer.tokenize(article['filename'] + '~' + article['body'])['input_ids'][0])
            if article_length >= 4096:
                problems.append((article['filename'],'ARTICLE_TOO_LONG: ' + str(len(article['headline']))))
        
        return problems
    
    def inference(self, articles : list,modulesToLoad=['*'], exclude_articles : list=[]):

        #Exclude news articles that we're not going to run inference through, this is
        #defined by the "exclude_articles" list, and whether we run it or not is
        #determined by an internal variable set at the __call__ method
        articles_to_exclude = []
        if self.exclude_problematic_articles:
            for i, article in enumerate(list(articles)):
                if article['filename'] in exclude_articles:
                    articles_to_exclude.append(i)
        articles_to_exclude.sort(reverse=True)
        for i in articles_to_exclude:
            _ = articles.pop(i)
        
        #Mark all articles as initially being TRUEs (drought-related) before passing them through the
        #binary classifier. This is done in case the user wants to run an
        #individual module and leave the binary classifier out. The code
        #below that performs the next steps in the pipeline, other than the
        #binary classification one, assumes that it will be given accurate
        #classification results by the binary model. If we're leaving
        #that functionality out, we'll simply mark them all as "positives",
        results_keyword = {i: 1 for i in range(len(articles))}
        results_binary = [1 for _ in range(len(articles))]
        
        #Boolean that guides whether we execute all modules in the pipeline (True) or not
        runAll = False

        #Check if the user has marked to run all modules or only some of them
        if len(modulesToLoad) == 1 and modulesToLoad[0] == '*':
            runAll = True

        #Apply keyword-based search to the articles.
        if runAll or 'keyword' in modulesToLoad:
            print("\nPerforming keyword-based classification")
            results_keyword = self.keyword(articles)

        #Binary classifier
        if runAll or 'binary' in modulesToLoad:
            print("\nPerforming binary classification")
            results_binary = self.binary_classifier(articles,results_keyword)
        
        #Gather a list of articles being labeled as positives, also keep alongside it the index to the original
        #list to know where it was located in the original corpus (via a tuple-based system)
        positives = [(i, articles[i]) for i, result in enumerate(results_binary) if result == 1]
        positives_idx = [positive_idx for positive_idx, _ in positives]

        #Separate all positive articles into its set of sentences. Although classification at the
        #sentence level is mostly used by the multiclass model, and hence it would make more sense
        #to make sentence splitting part of that system, we are experimenting with performing location NER at the
        #sentence level, so it makes more sense to store them separately to be retrieved at those two
        #different steps, so as to avoid repetition. Sentence splitting is done with spaCy. For each analyzed text,
        #we also store the spaCy Doc object that generated those sentences, since it could prove useful later on
        #for performing syntactic analysis for desambiguating toponyms.
        positives_sentences = []
        for positive_idx, positive_text in positives:
            sents, doc, sents_docs = self.sentence_split(positive_text)
            positives_sentences.append((positive_idx, sents, doc, sents_docs)) #format of individual entry: (index_int,[list of str],spacyDocObject)
        #positives_sentences = [(positive_idx, self.sentence_split(positive_text)) for positive_idx, positive_text in positives] #format of individual entry: (index_int,[list of str])
        positives_sentences_idx = [positive_idx for positive_idx, _, _, _ in positives_sentences]

        #Drought impacts classification
        impacts = defaultdict(list)
        if runAll or 'drought_impacts' in modulesToLoad:
            print("\nPerforming drought impacts classification")
            impacts = self.drought_impacts(positives_sentences)

        #NER locations
        locations = defaultdict(list)
        if runAll or 'ner_loc' in modulesToLoad:
            print("\nPerforming named entity recognition for places")
            for i, article_sentences, doc, sents_docs in tqdm(positives_sentences):
                toponyms, toponyms_metadata = self.ner_location(article_sentences,doc,sents_docs)
                if len(toponyms) > 0:
                    for j, toponym in enumerate(toponyms):
                        if toponym != '':
                            locations[i].append((toponym, toponyms_metadata[j]))
                """
                for sentence_num, sentence in enumerate(article_sentences):
                    toponyms, toponyms_metadata = self.ner_location([sentence],doc,[sents_docs[sentence_num]])
                    if len(toponyms) > 0:
                        for j, toponym in enumerate(toponyms):
                            if toponym != '':
                                locations[i].append((sentence_num, toponym, toponyms_metadata[j]))
                """

        #TO BE CONTINUED... TODO

        #Gather final results from all modules and export them in a list of dictionaries.
        #We can call some functions in the library to convert them to a readable format (ie CSV) later
        final_results = []

        for i, _ in enumerate(results_binary):
            cur_result = {
                'drought': bool(results_binary[i]) if runAll or 'binary' in modulesToLoad else None,
                'impacts': [],
                'locations': []
            }

            #Filename information
            if 'filename' in articles[i].keys():
                cur_result['filename'] = articles[i]['filename']
            else:
                cur_result['filename'] = ''

            #Drought impacts
            if '*' in modulesToLoad or 'drought_impacts' in modulesToLoad:
                if i in positives_idx:
                    cur_result['impacts'] = impacts[i]
            #NER location
            if '*' in modulesToLoad or 'ner_loc' in modulesToLoad:
                if i in positives_idx:
                    cur_result['locations'] = locations[i]
            
            #TODO
            #Write sentences and their index (use them in your output only if necessary)
            if i in positives_idx:
                cur_result['sentences_idx'] = dict()

                _, sents_final, _, _ = positives_sentences[positives_sentences_idx.index(i)]
                for sent_final_idx, sent_final in enumerate(sents_final):
                    cur_result['sentences_idx'][sent_final_idx] = sent_final

            final_results.append(cur_result)

        return final_results
    
    def __call__(self, path, isPath=True, modulesToLoad=['*'], exclude_problematic_articles=False):
        if isPath and not os.path.isdir(path):
            print("Path does not exist!")
            return
        if isPath:
            articles = load_articles_from_folder(path)
        else:
            articles = path
        
        problems = self.detect_problems_with_articles(articles)
        problems.extend(self.detect_repeated_articles(articles))

        self.problematic_articles = problems
        
        self.exclude_problematic_articles = exclude_problematic_articles

        #Make a list of the names for excluded articles; we don't include here those that have no headlines,
        #as although the headline for a specific article could be empty, the body of that article could still
        #contain worthwile information
        exclude_articles = list(set([article for article, problem in problems if problem != 'HEADLINE_TOO_SHORT']))
        
        return self.inference(articles,modulesToLoad,exclude_articles)

    def write_list_of_problematic_articles_to_file(self,filepath):
        with open(filepath,'w') as f:
            for problem in self.problematic_articles:
                f.write(problem[0] + '\t' + problem[1] + '\n')
        return

    """
    OUTPUT FUNCTIONS (FOR BETTER READABILITY OF RESULTS)
    """

    def write_impacts_to_csv_file(self,predictions,file):

        """
        OUTPUT FORMAT:
            Tab-separated values

            Example:
            Filename    Drought Agricultura Ganadería   Hídricos    Energético
            test.json   1   0   1   0   1
            test2.json  0   0   0   0   0
            test3.json  1   1   0   0   0
        """

        with open(file,'w',encoding='utf-8') as f:
            f.write('Filename\tDrought\tAgricultura\tGanadería\tHídricos\tEnergético\n')
            for prediction in predictions:
                impacts_str = ''
                filename = prediction['filename']
                drought = 1 if prediction['drought'] == True else 0

                f.write(filename + '\t' + str(drought) + '\t')

                if drought == 0:
                    f.write('0\t0\t0\t0\n')
                else:
                    impacts = prediction['impacts']

                    if 'Agricultura' in impacts:
                        f.write('1\t')
                    else:
                        f.write('0\t')

                    if 'Ganadería' in impacts:
                        f.write('1\t')
                    else:
                        f.write('0\t')

                    if 'Recursos hídricos' in impacts:
                        f.write('1\t')
                    else:
                        f.write('0\t')

                    if 'Energético' in impacts:
                        f.write('1')
                    else:
                        f.write('0')

                    f.write('\n')
        return

    def write_locations_to_text_file(self,predictions,file,include_and_simplify_coordinates=False):

        """
        OUTPUT FORMAT:
            Tab-separated values

            > WHEN 'include_and_simplify_coordinates' = False
            Example:
            Filename    Location    Type
            test1.json   location1   town
            test1.json   location2   dam
            test2.json   location1   UNK
            test2.json   location2   dam
            
            > WHEN 'include_and_simplify_coordinates' = True
            Example:
            Filename    Location    Type    Latitude    Longitude
            test1.json   location1   town    X   X
            test1.json   location2   dam    X   X
            test2.json   location1   UNK    0   0
            test2.json   location2   dam    X   X

        ||WARNING||:
        This function, when dealing with coordinates data expressed that is in polygons (e.g. that of a province area),
        it outputs the X/Y points of that polygon's centroid, in order to ease the storage and readability of data in a simple CSV file.
        If you wish to store the full polygon data, please use the alternative function called 'dump_toponyms_data_to_json_file'
        (although this will require you to process the data manually at a later step).
        """
        
        with open(file,'w',encoding='utf-8') as f:
            f.write('Filename\tLocation\tType\tLatitude\tLongitude\n')
            for prediction in predictions:
                for location, metadata in prediction['locations'].items():
                        f.write(prediction['filename'] + '\t' + location + '\t' + metadata['type'])
                        if include_and_simplify_coordinates:
                            if metadata['type'] != 'UNK':
                                f.write('\t' + str(metadata['coordinates_centroid_values']['latitude']) + '\t' + str(metadata['coordinates_centroid_values']['longitude']) + '\n')
                            else:
                                f.write('\t0\t0\n')
                        else:
                            f.write('\n')

        return
    
    def dump_toponyms_data_to_json_file(self,predictions,file):

        """
        OUTPUT FORMAT:
            JSON-serialized list of dictionaries
            [This function does NOT provide a 'pretty' print,
            but a complete dump of geographical data in its
            original complete format --> this includes Polygon-based
            data, which can be heavy to store; be careful!]

            Each dictionary in the list has the following format (example with one file that has two detected locations,
            one of them with polygon-based data):
            {
                'test1.json':
                {
                    'location1':
                    {
                        'type': 'town'
                        'coordinates':
                        {
                            'latitude': X,
                            'longitude': X
                        }
                    },
                    'location2':
                    {
                        'type': 'riv'
                        'coordinates': //GeoJSON object//
                    }
                }
            }
        """

        import json

        with open(file,'w',encoding='utf-8') as f:
            formatted_output_dictionary_list = {}
            for prediction in predictions:
                current_formatted_output_dictionary = {}
                
                for location, metadata in prediction['locations'].items():
                    current_formatted_output_dictionary[location] = {'type': metadata[type],
                                                                     'coordinates': metadata['coordinates']}

                formatted_output_dictionary_list[prediction['filename']] = current_formatted_output_dictionary

            json.dump(formatted_output_dictionary_list,f,ensure_ascii=False)

        return
    
