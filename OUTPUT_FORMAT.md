## Model's output format


Our framework outputs a list of Python dictionaries as the final result for each classification run. In that list, each dictionary corresponds exactly to one analyzed newspaper article. The resulting list can be serialized to a JSON file by the end user.

Due to the raw output format being quite complex because of the amount of encoded information, we will explain its format in detail in this file.

Each dictionary has the following series of keys:

- `drought`
- `impacts`
- `locations`
- `sentences_idx`

The first key, `drought`, tells the user if the current article is related to drought events or not. It is the output field for the binary classifier module. It stores a boolean variable with a `True` or `False` value --or a `None` type variable if the user skipped this step of the pipeline.

The next key, `impacts`, contains the series of detected drought impacts for the current article. It consists on a Python list composed of two main items. The first item is a list with a series of strings with the names of the detected drought impacts --in most cases, the end user will only be interested in this field. The list will be empty if no impacts were detected. The second item of the main list is another list, that contains the raw output logits for each of the sentences analyzed by each of the RoBERTa-based binary drought impacts classifiers. This additional information is only provided for debugging and for explainability purposes of the obtained results.

The `locations` key contains the found location names and its related metadata, such as geographical coordinates and the type of each toponym. The possible types of locations supported by our system, alongside the short names used by the model's output, can be found in the table below. This key contains a list with an item for each of the articl's sentences. For each sentence, there is a statically-sized list of two elements --as in the `impacts` key. The first item is a list of the named locations found in that sentence. The second item is another list which contains the metadata of the locations from the named locations list. For obtaining the metadata of a location that has an index `i` in the first list, the user has to access the `i`th element of this second list. This metadata is encoded as a Python dictionary, and it has the following keys: `start`, `end`, `coordinates` and `type`.

The first ones, `start` and `end`, contain numerical values with the 0-based starting and ending character indices of the current location in the sentence, as found by the NER module.

`Coordinates`, on the other hand, contains the geographical coordinate values for the current location --or is None if the location type is `UNK`. The output value types for this field depend on the location type for the current toponym. In the case of urban settlements, it is a Python dictionary with two self-explanatory keys: `latitude` and `longitude` (float numbers). If it's another entity, i.e. a broader area, its type corresponds to a PolyLine object from the Shapely Python library.

The `type` key corresponds to a short name-based description of the current found location, whose values are stated in the table below.

Finally, the `sentences_idx` key (short for 'Sentences index') consists of a Python dictionary in which each key-pair value corresponds to one of the individual sentences that the spaCy-based module has split the input text into. Each key has a 0-based index. The information in this field is only provided for debugging purposes, and thus can be ignored.


| Location type | Short name |
| --- | --- |
| Urban settlements (e.g. cities) |  `town`  | 
| Provinces |  `prov`  | 
| Autonomous communities |  `comm`  | 
| Rivers and river basins |  `riv`  | 
| Dams and reservoirs |  `dam`  | 
| Unknown (not found) |  `UNK`  | 
