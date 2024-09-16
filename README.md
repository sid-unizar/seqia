# SeqIA

Python-based framework for the automatic detection of drought-related news and their impacts. It is based on the use of several NLP libraries, including transformers and spaCy.

GPU use strongly advised.


## INSTALLATION

1- Create a Python virtual environment

Under Linux:

```
virtualenv venv -p python3.6
source venv/bin/activate
```

Under Windows:

```
python -m venv c:\path\to\myenv

c:\path\to\myenv\Scripts\activate
```

2- Then run the following command:

```
pip install spacy
python -m spacy download es_core_news_sm
```

3- Run the following Bash script (Linux; Windows instructions below):

```
bash download_geonames.sh
```

This will automatically download a dataset file that contains a series of coordinates and toponyms from the Geonames online platform.

If you're running under Windows, simply manually download the following ZIP file:
[https://download.geonames.org/export/dump/ES.zip](https://download.geonames.org/export/dump/ES.zip)

Then uncompress it and place its contents inside the following folder:

```
PATH_TO_THIS_REPOSITORY\seqia\geonames\ES\
```

We have also included a PowerShell-based script (```download_geonames.ps1```) in case you want to do the process automatically under Windows. Your system will need to be able to run PowerShell scripts in order to work:

```
powershell.exe -noexit "& '.\download_geonames.ps1'"
```

4- **Manually** download the following files from these links:

<https://www.mapama.gob.es/app/descargas/descargafichero.aspx?f=RiosCompPfafs.kmz>

<https://www.mapama.gob.es/app/descargas/descargafichero.aspx?f=egis_embalse_geoetrs89.kmz>

These files are provided by Ministerio para la Transicion Ecologica (Spanish Ministery for Ecological Transition; or MiTEco for short), and cannot be automatically fetched via scripts due to the service imposing limitations in this regard.
Once downloaded, you will have to copy both files to the following directory:

```
PATH_TO_THIS_REPOSITORY\seqia\loc_files\
```

5- **VERY IMPORTANT:** The models' weights have not been included within the actual repo because of space issues. They are available from this repository, however, in the Releases tab:

### Manual installation

5.1- The latest release in this Github repository contains the model weight files. Each of the files for each of the models has been renamed to easily identify them. You'll have to manually download them and place them inside their respective folders. More instructions on this procedure can be found in the Latest release README section.

5.2- Once the files are in their respective directories, RENAME them to "pytorch_model.bin"

### Automatic installation (Linux-only; requires GitHub CLI)

Optionally, under Linux, if you have access to this repository and you have set up [GitHub CLI](https://cli.github.com/) in your machine, you can run the following Bash script instead:

```
bash download_model_weights.sh
```

This will do the same process as above, but automatically.

6- Finally, run ```pip install -e .``` to install this library


## USE

The functionality of this library is implemented inside a main class called `DroughtClassifier`, of which you first have to create an instance. This is how it can be used:

```
from seqia import DroughtClassifier

classifier = DroughtClassifier()

predictions = classifier(path_to_folder_with_jsons)
```

Once you obtain these predictions, you can output them to a human readable format via the use of the following available support functions:


- Export drought impacts to a CSV file:
```
#NOTE: variable "predictions" is obtained in a prior step
classifier.write_impacts_to_csv_file(predictions,path_to_a_csv_file)

```

- Export detected location names to a CSV file:
```
classifier.write_locations_to_csv_file(predictions,path_to_a_csv_file)

```

- (*Advanced users only*) Export detected location names and geographical coordinates to a JSON file (more complex, can lead to higher output file sizes):
```
classifier.dump_toponyms_data_to_json_file(predictions,path_to_a_json_file)

```

**NOTE:** These functions process the raw output format as provided by the model, but exclude internal information on the process; such data is not necessary for obtaining relevant results, but if the end user wishes to analyze it, and extract some information in the process that the functions above might omit, we refer them to the file `OUTPUT_FORMAT.MD` located in this same repository.

# ADDITIONAL FUNCTIONALITY

## Automatic detection of empty and badly-formatted articles

When you call the main class instance in order to run inference, you can pass in an additional parameter to the function, `exclude_problematic_articles=True`, which will run some optional checks to ensure that the passed-in corpus does not have some issues like: repeated articles, empty news articles... The function can be called in the following way:

```
predictions = classifier(path_to_folder_with_jsons, exclude_problematic_articles=True)
```

Once the inference stops running, you can also call a function that will allow you to dump to a CSV file all the excluded JSON files, alongside a reason for why they were excluded. This can be done via this function call:
```
classifier.write_list_of_problematic_articles_to_file('path/to/csv_file.csv)
```

Some of the reasons for the exclusion of articles can be the following:

| Short name  | Description |
| ------------- | ------------- |
| 'REPEATED_ARTICLE_BODY' | There is a duplicate article (shares the same article body)  |
| 'BODY_TOO_SHORT'  | The article body is less than three characters long  |
| 'HEADLINE_TOO_SHORT'  | The article headline is empty (will NOT be excluded by default from being run in inference) |
| 'ARTICLE_TOO_LONG'  | The passed-in article's length, in RoBERTa BPE tokens, is bigger than the 4,096 maximum tokens a Longformer-based model can accept.  |

The list of currently excluded articles is kept as long as you don't run the inference function again!

## Run only selected parts of the pipeline

The seqia library is implemented in a series of separate steps, part of a pipeline that gathers raw text data from JSON-based articles and outputs a series of other JSON files that contain information on whether the passed-in corpus has drought-related articles and their impacts (if any).

However, for debugging purposes, you may wish to skip some of those parts and only run, for instance, Named Entity Recognition for places or skip binary classification. For that, when you obtain your created instance of the main ```DroughtClassifier``` class, you can pass in an optional parameter called ```modulesToLoad```, which accepts a List with a series of strings of the pipeline steps to run. These strings can be set to a combination of the "short name" values found in the table below:

| Short name  | Pipeline step |
| ------------- | ------------- |
| * | Run ALL pipeline steps (default value)  |
| 'keyword'  | Binary classification (drought/not drought) **based on the use of keyword-based searches**. |
| 'binary'  | Binary classification (drought/not drought; Transformer-based)  |
| 'drought_impacts'  | Drought impacts classification |
| 'ner_loc'  | Named Entity Recognition (NER) for locations  |

Then you can call the main function in the following way:
```
from seqia import DroughtClassifier

steps = ['binary', 'ner_loc'] #Only the steps listed in this variable will be run

classifier = DroughtClassifier()
predictions = classifier(path_to_folder_with_jsons, modulesToLoad=steps)
```

## Use CPU in inference and options

If your machine does not have a GPU for running inference, the library will automatically detect it and run inference in the available CPUs. A display warning will be shown when the library is run only in CPU mode:

```
WARNING! Running inference in CPU. You should better use a GPU.
```

### Limit use of CPU threads

By default the library will attempt to use as many CPU cores as the machine has available. However, if you wish to keep the CPU usage down and use less threads, you can pass an optional parameter, called ```cpu_threads``` (which accepts an ```int```), when you create an instance of the ```DroughtClassifier``` class. The parameter in that function controls the number of CPU threads the library will make use of:

```
from seqia import DroughtClassifier

classifier = DroughtClassifier(cpu_threads=48)  #Will use 48 CPU threads
```

You can change this parameter after creating an instance of the object, by calling an object's method known as ```this->change_number_cpu_threads(num)```, where ```num``` is the number of CPU threads you wish to use:

```
from seqia import DroughtClassifier

classifier = DroughtClassifier()

classifier.change_number_cpu_threads(48)  #Will use 48 CPU threads
```
