

"""
MÓDULO PARA LA DETECCIÓN DE LUGARES GEOGRÁFICOS DE CARÁCTER HIDROLÓGICO EN TEXTOS

Requiere:
    - transformers (pipeline)

Hace uso de un modelo fine-tuneado para NER, entrenado por el BSC con datos de la BNE, y disponible en HuggingFace:
  https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus

El modelo base comete fallos a la hora de asignar algunas etiquetas IOB a los nombres de lugar, por lo que nuestro código
intercepta algunos de esos fallos e intenta corregirlos. Sin embargo, es posible que siga habiendo falsos positivos,
errores...
"""

from collections import defaultdict
from transformers import pipeline
import os
import geopandas
import pandas as pd
import shapely
import xml.etree.ElementTree as ET
import zipfile

class NERLocation:
  #Huggingface variables
  model_name = "PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus"
  MODEL_MAX_SIZE = 512

  ##################
  ## Constructor ##
  #################
  def __init__(self, device):

    #Load pipe
    self.pipe = pipeline("token-classification", model=self.model_name, device=device)

    #Load offline localization data from IGN and Geonames
    self.load_localization_data()
    
    self.preload_misc_geolocation_variables()

    self.do_geocoding = True

    return
  
  #############################
  ## Loading data functions ##
  ############################
  """
  The functions defined below load a series of offline files
  located in a folder from the repository. These files contain
  a series of geographical coordinate points for several types
  of toponyms, such as towns, rivers, dams...
  """
  def load_localization_data(self):

    self.towns = self.load_towns_data()
    self.countries = self.load_countries_data()
    self.comm, self.alt_comm_names = self.load_autonomous_communities_data()
    self.prov, self.alt_prov_names = self.load_provinces_data()
    self.riv = self.load_rivers_data()
    self.dams = self.load_dams_reservoirs_data()

    self.alt_prov_names.update({'Vizcaya': 'Bizkaia',
                                'Guipúzcoa': 'Gipuzkoa',
                                'Islas Baleares': 'Illes Balears',
                                'Baleares': 'Islas Baleares',
                                'Coruña': 'A Coruña',
                                'Lérida': 'Lleida'
    })
    
    self.alt_comm_names.update({'Asturias': 'Principado de Asturias',
                                'Baleares': 'Illes Balears', 'Islas Baleares': 'Illes Balears',
                                'Islas Canarias': 'Canarias',
                                'Castilla La Mancha': 'Castilla-La Mancha',
                                'Comunidad Valenciana': 'Comunitat Valenciana',
                                'Navarra': 'Comunidad Foral de Navarra',
                                'Madrid': 'Comunidad de Madrid'})

    self.rivers_exceptions = ['España', 'Francia']
    
    self.communities_with_shared_capital_city_name = ['Madrid', 'Murcia', 'Ceuta', 'Melilla']

    self.geonames = self.load_geonames_data()

    return

  def load_towns_data(self):
    
    towns = dict()
    repeated_towns = defaultdict(list)

    with open(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),'MUNICIPIOS.csv'),'r',encoding='cp1252') as f:
      for i, line in enumerate(f):
        if i == 0:
          continue
        if line[-1] == '\n':
          line = line[:-1]
        
        line = line.split(';')
        town_names = line[4].split('/')
        for town_name in town_names:
          if town_name not in towns.keys():
            towns[town_name] = {'longitude': float(line[12].replace(',','.')), 'latitude': float(line[13].replace(',','.'))}
          else:
            repeated_towns[town_name].append({'longitude': float(line[12].replace(',','.')), 'latitude': float(line[13].replace(',','.'))})

    with open(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),'other_towns.tsv'),'r',encoding='utf-8') as f:
      for line in f:
        if line[-1] == '\n':
          line = line[:-1]
        
        town_names, coords = line.split('\t')

        town_names = town_names.split('/')

        coords = coords.split(' ')

        for town_name in town_names:
          if town_name not in towns.keys():
            towns[town_name] = {'latitude': float(coords[1][:-1]), 'longitude': float(coords[0][6:])}
          else:
            repeated_towns[town_name].append({'latitude': float(coords[1][:-1]), 'longitude': float(coords[0][6:])})

    towns['REPEATED_TOWNS'] = repeated_towns

    return towns

  def load_provinces_data(self):
  
    prov_alt_names = dict()
    prov = geopandas.read_file(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),'au_AdministrativeUnit_3rdOrder0.gml'))

    prov_names = prov['text'].to_list()

    for names in prov_names:
      for name in names.split('/'):
        prov_alt_names[name] = names

    return prov, prov_alt_names
  
  def load_autonomous_communities_data(self):

    comm_alt_names = dict()
    comm =  geopandas.read_file(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),'au_AdministrativeUnit_2ndOrder0.gml'))

    comm_names = comm['text'].to_list()

    for names in comm_names:
      for name in names.split('/'):
        comm_alt_names[name] = names

    return comm, comm_alt_names

  def parse_KML_miteco_file(self,filepath):

      #Parses the KML file format provided by Ministerio para la Transicion Ecologica (MiTEco)
      #For loading this file, we've written our own custom parser for KML files. This file format uses
      #XML, so it's just a matter of iterating over each individual XML entry and finding the appropriate information.
      
      tree = ET.parse(filepath)
      root = tree.getroot()
      
      df_tmp = []

      for entry in root:
        for child in entry:
          for placemark in child:
            if placemark.tag == '{http://www.opengis.net/kml/2.2}Placemark':
              skip = False
              cur_entry = {}
              for item in placemark:
                #River name
                if item.tag == '{http://www.opengis.net/kml/2.2}name':
                  if item.text.split(' - ')[0].strip() == 'SIN NOMBRE':
                    skip = True
                    break   #Skip untitled entries
                  name = item.text.split(' - ')[0].strip()
                  name = name.split(',')
                  if len(name) > 1:
                    if name[1].lower() == 'l\'':
                      name = name[1].strip() + name[0]
                    else:
                      if not name[0].startswith(name[1].strip() + ' '):
                        name = name[1].strip() + ' ' + name[0]
                      else:
                        name = name[0]
                  else:
                    name = name[0]
                  name = name.split('(')[0]
                  cur_entry['text'] = name
                
                #Polylines
                elif item.tag == '{http://www.opengis.net/kml/2.2}LineString':
                  for linestring_child in item:
                    if linestring_child.tag == '{http://www.opengis.net/kml/2.2}coordinates':
                      coords = []
                      coords_tmp = linestring_child.text.split(' ')
                      for coord_triple in coords_tmp:
                        coord_triple = coord_triple.split(',')
                        coords.append([float(coord_triple[0]), float(coord_triple[1])])
                      
                      cur_entry['geometry'] = shapely.LineString(coords)  #shapely.geometry.polygon.Polygon(coords)
                      
                      if not skip:
                        df_tmp.append(cur_entry)
                        skip = False
                
                elif item.tag == '{http://www.opengis.net/kml/2.2}Polygon':
                  for multigeo_child in item:
                    if multigeo_child.tag == '{http://www.opengis.net/kml/2.2}outerBoundaryIs':
                      for linestring_child in multigeo_child:
                        if linestring_child.tag == '{http://www.opengis.net/kml/2.2}LinearRing':
                          for coords_child in linestring_child:
                            if coords_child.tag == '{http://www.opengis.net/kml/2.2}coordinates':
                              coords = []
                              coords_tmp = coords_child.text.split(' ')
                              for coord_triple in coords_tmp:
                                coord_triple = coord_triple.split(',')
                                coords.append([float(coord_triple[0]), float(coord_triple[1])])

                              cur_entry['geometry'] = shapely.LineString(coords)

                              if not skip:
                                df_tmp.append(cur_entry)
                                skip = False

                elif item.tag == '{http://www.opengis.net/kml/2.2}MultiGeometry':
                  for multigeo_child in item:
                    if multigeo_child.tag == '{http://www.opengis.net/kml/2.2}LineString':
                      for linestring_child in multigeo_child:
                        if linestring_child.tag == '{http://www.opengis.net/kml/2.2}coordinates':
                          coords = []
                          coords_tmp = linestring_child.text.split(' ')
                          for coord_triple in coords_tmp:
                            coord_triple = coord_triple.split(',')
                            coords.append([float(coord_triple[0]), float(coord_triple[1])])

                          cur_entry['geometry'] = shapely.LineString(coords)

                          if not skip:
                            df_tmp.append(cur_entry)
                            skip = False
              
              #Append curEntry to DF (if it's not an empty name)
              #if not skip:
              #  df_tmp.append(cur_entry)
              #  skip = False

      return pd.DataFrame().from_dict(df_tmp)

  def load_rivers_data(self):
    
    #Loads river names and geometry data from a file provided by Ministerio para la Transicion Ecologica (MiTEco)
    #The data is provided in a KMZ format, a compressed archive (Zip format) which contains a series of KML files.
    #We've written a custom parser for KML files in a function above, which we use here.
    #NOTE: The commented-out line at the end refers to an older resource that we used, originating from
    #Instituto Geografico Nacional (IGN), but which is much less complete and that we discarded.
    
    kml_files = ['A_RiosCompletosv2.kml','M_RiosCompletosv2.kml','Ca_RiosCompletosv2.kml']

    kml_dfs = []

    #The KML files are found inside a compressed KMZ (the final "Z" stands for ZIP). The compressed archive
    #is over 1GB in size, but if uncompressed it would have a much bigger footprint! As such, we keep the file compressed within
    #our repository directory, and only uncompress it on the go. For that, we uncompress each KML file inside the KMZ to a temp
    #file, read it and we later get rid of that temporary file
    with zipfile.ZipFile(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),'rioscomppfafs.kmz'), 'r') as myzip:
      for kml in kml_files:
        with myzip.open(kml) as myfile:
          tmp_file_path = os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),kml)
          with open(tmp_file_path,'wb',encoding='utf-8') as fp:
            fp.write(myfile.read())
            fp.seek(0)
          
          kml_dfs.append(self.parse_KML_miteco_file(tmp_file_path))

          os.remove(tmp_file_path)

    return pd.concat(kml_dfs,ignore_index=True)
    #return geopandas.read_file(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),'hy-p_RiverBasin0.gml'))

  def load_dams_reservoirs_data(self):

    file = 'egis_embalse_geoetrs89'#, 'egis_presa_geoetrs89']

    with zipfile.ZipFile(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),file + '.kmz'), 'r') as myzip:
      with myzip.open(file + '.kml') as myfile:
        tmp_file_path = os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),file + '.kml')
        with open(tmp_file_path,'wb',encoding='utf-8') as fp:
          fp.write(myfile.read())
          fp.seek(0)

        df = self.parse_KML_miteco_file(tmp_file_path)

        os.remove(tmp_file_path)
        
    return df

  def load_geonames_data(self):
    geonames = defaultdict(list)
    geonames_alternative_names_index = defaultdict(list)
    geonames_raw = []
    with open(os.path.join(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'geonames')),'ES'),'ES.txt'),'r',encoding='utf-8') as f:
      geonames_raw = f.read()
      geonames_raw = geonames_raw.splitlines()

    if len(geonames_raw) <= 0:
      print("Geonames database not found!")
      return None

    for line in geonames_raw:
      
      #Sometimes variants of a towns' spelling are listed out in a same entry using a slash
      #to differentiate them. As in "Donostia/San Sebastián". Simply split the string by
      #the slash and then add both variants of the same name to our database as distinct entries.
      line = line.split('\t')
      names = line[1].split('/')
      
      for name in names:
        #Handling of a small exceptional case: there is a town in Catalonia called "Iran",
        #similar to the Middle East country, only that without an accent. This is not
        #a very elegant solution, but we simply replace the affected name here
        if name == 'Irán':
          name = 'Iran'

        skip = False
        #Handle entries in Geonames formatted like this: "Lorcha/Orxa, l'"
        #Convert "Lorcha/Orxa, l'" to ['Lorcha', 'l\'Orxa']
        #Also adds an additional entry to the database without a prepending article: "Tres Villas" vs. "Las Tres Villas" and "Orxa" vs. "l'Orxa"
        name = name.split(',')
        if len(name) > 1:
          name_without_article = name[0]
          if name[1].lower().strip() == 'l\'':
            name = name[1].strip() + name[0]
          else:
            if not name[0].startswith(name[1].strip() + ' '):
              name = name[1].strip() + ' ' + name[0]
            else:
              name = name[0]
              skip = True
          
          curEntry = {}
          #line = line.split('\t')
          curEntry['geonameid'] = line[0]
          curEntry['name'] = name_without_article
          curEntry['asciiname'] = line[2]
          alternateNames = line[3].split(',')
          for alt_name in alternateNames:
            geonames_alternative_names_index[alt_name].append(curEntry['name'])
        
          curEntry['alternatenames'] = alternateNames
          curEntry['latitude'] = line[4]
          curEntry['longitude'] = line[5]
          curEntry['feature class'] = line[6]
          curEntry['feature code'] = line[7]
          curEntry['country code'] = line[8]
          curEntry['cc2'] = line[9].split(',')
          geonames[curEntry['name']].append(curEntry)
        else:
          name = name[0]
        
        curEntry = {}
        #line = line.split('\t')
        curEntry['geonameid'] = line[0]
        curEntry['name'] = name
        curEntry['asciiname'] = line[2]
        alternateNames = line[3].split(',')
        for alt_name in alternateNames:
          geonames_alternative_names_index[alt_name].append(curEntry['name'])
      
        curEntry['alternatenames'] = alternateNames
        curEntry['latitude'] = line[4]
        curEntry['longitude'] = line[5]
        curEntry['feature class'] = line[6]
        curEntry['feature code'] = line[7]
        curEntry['country code'] = line[8]
        curEntry['cc2'] = line[9].split(',')
        #...
        
        if not skip:
          geonames[curEntry['name']].append(curEntry)
        else:
          skip = False
        
    for k, v in geonames_alternative_names_index.items():
      geonames_alternative_names_index[k] = list(set(v))

    geonames['ALTERNATIVE_NAMES'] = geonames_alternative_names_index

    return geonames
  
  def load_countries_data(self):

    countries = dict()

    with open(os.path.join((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'loc_files')),'countries.csv')) as f:
      for i, line in enumerate(f):

        if i == 0:
          continue

        if line[-1] == '\n':
          line = line[:-1]

        line = line.split('\t')

        latitude = line[1].replace('\U00002013', '-').replace(',','.')
        longitude = line[2].replace('\U00002013', '-').replace(',','.')

        if latitude == '' or longitude == '':
          continue

        countries[line[3]] = {'latitude': float(latitude), 'longitude': float(longitude)}

    return countries

  ##################################
  ## NER TOKENS AGGREGATION CODE ##
  #################################

  def loc_tokens_aggregation(self,predicted_token_class,text):
    #The output of the HuggingFace model returns a list of dictionaries
    #with a set of I-O-B tags, which are not aggregated by default (that is,
    #they are not joined together, they're presented as individual entries).
    #Although the HuggingFace APIs do allow for an automatic aggregation to
    #be done straight from the pipeline interface, the output for this NER model
    #is imperfect: for instance, it sometimes assigns an "E-" (ending)
    #tag right after an "S-" (single) tag, WHICH SHOULD NOT TECHNICALLY
    #BE PERMITTED. However, since it is an statistical model, these kinds
    #of errors are to be expected. As a result, we have manually examined
    #the output of many examples of the NER model, and have come up with some
    #manual aggregation strategies that attempt to solve some of these issues.
    #The code below not only allows for the regular, expected aggregation of well-formatted
    #IOB tags, but it also accounts for some cases where we observed common
    #occurrences of formatting mistakes that could be identified and generalized.
    #Please take in note that some mistakes are still to be expected.
    prior_IOB_tag = ''

    toponyms = []
    toponyms_metadata = []

    #Iterate over each predicted text
    for text_num, pred_text in enumerate(predicted_token_class):

      #Reset variables
      curTopList = []
      curTopMetadataList = []
      curTop = ''
      curTopMetadata = {}

      #Iterate over each token of each individual predicted text
      #Do manual token aggregation using some custom code, since
      #the predicted labels are sometimes not well assigned
      for i, ent in enumerate(pred_text):

        space = ''  #Reset variable

        pred = ent['entity']
        start = ent['start']
        end = ent['end']

        if pred.endswith('LOC'):
          if pred.startswith('B-'):

            if prior_IOB_tag == 'S-':
              if len(curTop) > 1:
                curTopList.append(curTop)
                curTopMetadataList.append(curTopMetadata)
              curTopMetadata = {}
            elif prior_IOB_tag == '(S)E-':
              if len(curTop) > 1:
                curTopList.append(curTop)
                curTopMetadataList.append(curTopMetadata)
              curTopMetadata = {}

            curTop = text[text_num][start:end]

            curTopMetadata = {'start': start, 'end': end}
            prior_IOB_tag= 'B-'
          
          elif pred.startswith('I-'):
            
            if 'end' in curTopMetadata.keys() and curTopMetadata['end'] != start:
              space = ' '

            curTopMetadata['end'] = end
            curTop += space + text[text_num][start:end]
            prior_IOB_tag = 'I-'

          elif pred.startswith('E-'):

            if 'end' in curTopMetadata.keys() and curTopMetadata['end'] != start:
              space = ' '

            if i == 0:  #Weird case: the model has assigned the very first token in the list with an "E" tag...
              curTop = text[text_num][start:end]
              curTopMetadata = {'start': start, 'end': end}
              prior_IOB_tag = '(S)E-'
              continue
            
            if prior_IOB_tag == 'S-':
              curTop += space + text[text_num][start:end]
              curTopMetadata['end'] = end
              prior_IOB_tag = '(S)E-'

              if i == len(pred_text)-1:
                if len(curTop) > 1:
                  curTopList.append(curTop)
                  curTopMetadataList.append(curTopMetadata)
                curTopMetadata = {}
                continue
            
            elif prior_IOB_tag == 'B-':
              curTop += space + text[text_num][start:end]
              curTopMetadata['end'] = end
              prior_IOB_tag = '(S)E-'

              if i == len(pred_text)-1:
                curTopMetadata['end'] = end
                if len(curTop) > 1:
                  curTopList.append(curTop)
                  curTopMetadataList.append(curTopMetadata)
                curTopMetadata = {}
                continue

            elif prior_IOB_tag == '(S)E-':
              curTop += space + text[text_num][start:end]
              curTopMetadata['end'] = end
              prior_IOB_tag = '(S)E-'

              if i == len(pred_text)-1:
                if len(curTop) > 1:
                  curTopList.append(curTop)
                  curTopMetadataList.append(curTopMetadata)
                curTopMetadata = {}
                continue

            else:
              curTop += space + text[text_num][start:end]
              curTopMetadata['end'] = end
              prior_IOB_tag = '(S)E-'

          elif pred.startswith('S-'):
            
            if prior_IOB_tag == 'S-' or prior_IOB_tag == '(S)E-':
              if 'end' in curTopMetadata.keys() and curTopMetadata['end'] == start:
                #Two consecutive "S" tags, but they are adjacent in the text: that
                #means they are actually part of a single tag!
                curTop += text[text_num][start:end]
                curTopMetadata['end'] = end
                prior_IOB_tag = '(S)E-' #'S-'

                if i == len(pred_text)-1:
                  if len(curTop) > 1:
                    curTopList.append(curTop)
                    curTopMetadataList.append(curTopMetadata)
                  curTopMetadata = {}
                  continue
                continue

              else:
                if len(curTop) > 1:
                  curTopList.append(curTop)
                  curTopMetadataList.append(curTopMetadata)
                curTopMetadata = {}
            elif prior_IOB_tag == '(S)E-':
              curTopList.append(curTop)
              curTopMetadataList.append(curTopMetadata)
              curTopMetadata = {}
            
            curTop = text[text_num][start:end]
            curTopMetadata = {'start': start, 'end': end}
            prior_IOB_tag = 'S-'

            if i == len(pred_text)-1:
              if len(curTop) > 1:
                curTopList.append(curTop)
                curTopMetadataList.append(curTopMetadata)
              curTopMetadata = {}
              continue

      #Add found toponyms of current text to list of toponyms
      toponyms.append(curTopList)
      toponyms_metadata.append(curTopMetadataList)

    return toponyms, toponyms_metadata
  
  ########################
  ## PRE-LOAD FUNCTION ##
  #######################

  def preload_misc_geolocation_variables(self):
    #Precomputes some variables used repeteadly by the geolocation function below.
    #This will hopefully speed up execution a bit...

    self.town_names = list(self.towns.keys())
    #self.town_names_with_variants = list(self.a.keys())

    self.country_names = list(self.countries.keys())

    self.comm_names = list(self.comm['text'].to_list())
    self.comm_names_with_variants = list(self.alt_comm_names.keys())

    self.riv_names = list(self.riv['text'].to_list())

    self.dam_names = list(self.dams['text'].to_list())

    self.prov_names = list(self.prov['text'].to_list())
    self.prov_names_with_variants = list(self.alt_prov_names.keys())

  ###########################
  ## GEOLOCATION FUNCTION ##
  ##########################

  def geolocation_IGN(self,toponyms,toponyms_metadata,doc,doc_sentences,simplifyPolylines=False):
    #Function that matches all found toponyms with a series of
    #coordinate values originating from the IGN database.
    #It also identifies the type of toponym it talks about (river,
    #town...).
    #This function also leaves out toponyms that are located
    #outside of Spain, via a really simple idea: the used local databases
    #use information originating only from Spain :-). That way, we can
    #ensure that we do not get an instance of "Guadalajara (México)"
    #instead of "Guadalajara" in Spain.

    for i, toponyms_list in enumerate(toponyms):
      doc_sentence = doc_sentences[i]
      for j, toponym in enumerate(toponyms_list):
        
        #Pre-fill metadata structure with some fields for type of toponym and its coordinates
        toponyms_metadata[i][j]['coordinates'] = None
        toponyms_metadata[i][j]['type'] = ''

        #This list stores the candidate type of toponym the current proper name could
        #be talking about, and where within the database it can be found
        found = ['','']

        #Misc variables
        ambRef = False
        toponym_uppercased = toponym.upper().replace('Á','A').replace('É','E').replace('Í','I').replace('Ó','O').replace('Ú','U').strip()

        #Try to see if the current entry is found as-is within the names database of IGN

        #1-Towns
        if toponym in self.town_names:
          if found[0] == '' and toponym != 'Irán':
            found = ['town',self.town_names.index(toponym)]
            if toponym in self.towns['REPEATED_TOWNS'].keys() or toponym == 'La Palma': #There is a "La Palma" in Murcia and in Canarias
              #There are two towns WITH THE SAME NAME in the database
              #One example is "Alcolea", the name of a village both in Córdoba and in Almería
              #Append a third item in the "found" list, which simply contains the name of
              #the current toponym. This variable could really be anything: the important
              #thing is for there to be a third item in the list to serve as a flag to be used by the
              #code below to check for repeated entries.
              found.append(toponym)

        #2-Autonomous communities
        if toponym in self.comm_names_with_variants:
          if found[0] == '':
            found = ['comm', '']
          elif found[0] != '':
            #Ambiguous reference: toponyms such as "Madrid", "Valencia" or "Murcia" can refer to either a city or its surrounding autonomous community
            #Disambiguate this reference to see what it actually refers to
            
            #TODO: For the moment being, we assign it as being related to the capital city ONLY, but we need to come up with more proper disambiguation strategies
            if toponym in self.communities_with_shared_capital_city_name:
              found = ['town', self.town_names.index(toponym)]
            else:
              found = ['town', self.town_names.index(toponym)]
          
          if toponym == 'Aragón':
            #Handling of an exceptional case: Aragón can be both the name of the autonomous community and that of a river ("Aragón" vs. "Río Aragón")
            #Run the toponym through a disambiguation function. By default, the function will return the toponym type "town" if it's not
            #a river, hence the weird check done below
            if self.disambiguate_toponym_type(toponym,doc_sentence) == 'town':
              found = ['comm', '']
            else:
              found = ['riv', self.riv_names.index('RIO ARAGON')]
          
          if found[0] == 'comm':
            if toponym in self.comm_names:
              found[1] = self.comm_names.index(toponym)
            else:
              found[1] = self.comm_names.index(self.alt_comm_names[toponym])
        
        #3-Provinces
        if toponym in self.prov_names_with_variants:
          if found[0] == '':
            found = ['prov', '']
          elif found[0] == 'town':
            #Ambiguous name: it is both the name of a province and a city (ex.: "provincia de Zaragoza" vs. "Zaragoza").
            
            ambRef = True   #Boolean that is set when it is an ambiguous reference

            #Step 1) Check if the toponym is enclosed by parantheses (ex: "(Zaragoza)"): 90% of the times it will be a province
            is_candidate_province = False
            for token in doc_sentence:
              if token.text == toponym:
                for left in token.lefts:
                  if left.text == '(' and left.tag_ == 'PUNCT' and left.idx == (token.idx-1):
                    is_candidate_province = True
                    break
                
                if is_candidate_province:
                  for right in token.rights:
                    if right.text == ')' and left.tag_ == 'PUNCT' and right.idx == (token.idx + len(token.text)):
                      is_candidate_province = False
                      ambRef = False  #ambRef = False if condition is succesful
                      found[0] = 'prov'
                      break
            
            #Step 2) Run this toponym through our toponym type disambiguation function, which will try to check if the toponym is preceeded
            #by the keyword "provincia(s)". If it doesn't find that reference, it will default to identifying it as a town.
            if ambRef:
              found[0] = self.disambiguate_toponym_type(toponym,doc_sentence)
              ambRef = False
          
          elif found[0] == 'comm':
            #Ambiguous name: it is both the name of a province and an autonomous community.
            #SOLUTION: This only happens with communities with a single province. As a result, keep it at the
            #level of the autonomous community
            pass

          if found[0] == 'prov':
            if toponym not in self.prov_names:
              found[1] = self.prov_names.index(self.alt_prov_names[toponym])
            else:
              found[1] = self.prov_names.index(toponym)

        #4-Rivers
        if found[0] == '' and toponym_uppercased in self.riv_names and toponym not in self.rivers_exceptions:  #There is a "Río España" in Asturias, omit this reference ONLY if it's not prepended by the keyword "río"
          found = ['riv', self.riv_names.index(toponym_uppercased)]
        
        #4.1-Rivers without prepended "RIO" keyword (for example, "Tajo", without "río Tajo")
        if found[0] == '' and 'RIO ' + toponym_uppercased.strip() in self.riv_names:
          if toponym not in self.rivers_exceptions:
            found = ['riv', self.riv_names.index('RIO ' + toponym_uppercased.strip())]
          else:
            #There are some rivers in Spain called "Río España" and "Río Francia". These entries appear in our local databases.
            #If we simply matched any appeareance of the words "España" or "Francia" with either of these entries, we would match
            #98% of the times the name of each country to an unrelated Spanish river. In order to avoid these false positives, yet
            #still keeping a small check to see if we could potentially be talking about these specific rivers, we run that toponym
            #through the disambiguation function to see if it's preceded by the keyword "río". Then, AND ONLY THEN, we will assume
            #it's talking about the rivers, not the countries.
            if self.disambiguate_toponym_type(toponym,doc_sentence) == 'riv':
              found = ['riv', self.riv_names.index('RIO ' + toponym_uppercased.strip())]
            else:
              if toponym in self.country_names:
                found = ['country', toponym.strip()]
        
        #5-Countries
        if found[0] == '' and toponym in self.country_names:
          found = ['country', self.country_names.index(toponym.strip())]
        
        #6-Dams/reservoirs
        if toponym_uppercased.strip() in self.dam_names:
          toponym_type = self.disambiguate_toponym_type(toponym,doc_sentence)
          if toponym_type == 'dam':
            found = ['dam', self.dam_names.index(toponym_uppercased.strip())]

        if found[0] == '' and toponym_uppercased.strip() in self.dam_names:
          found = ['dam', self.dam_names.index(toponym_uppercased.strip())]

        if found[0] == 'town' or found[0] == 'comm' or found[0] == 'prov' or found[0] == 'riv' or found[0] == 'dam' or found[0] == 'country':
          #Yes...

          #First, try to locate toponyms that are shared across
          #rivers and towns: for instance, Turia and Río Turia.
          #For this, we search the current toponym preprended by
          #the keyword "Río ", and we look up if there's an
          #additional entry within the IGN database.
          if found[0] != 'riv' and found[0] != 'country' and "RIO " + toponym_uppercased.strip() in self.riv_names and toponym != 'Aragón' and toponym not in self.rivers_exceptions:
            #If a name is shared across these two IGN entries, run a 
            #function that attempts to disambiguate the current
            #toponym's type based on some linguistic cues.
            toponym_type = self.disambiguate_toponym_type(toponym,doc_sentence)
            
            if toponym_type == 'riv':
              #River
              idx = "RIO " + toponym_uppercased.strip()
              if not simplifyPolylines:
                toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.riv.loc[self.riv['text'] == idx].iloc[0]['geometry'])
                
              else:
                geometry = self.riv.loc[self.riv['text'] == idx].iloc[0]['geometry'].centroid
                toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                
              geometry = self.riv.loc[self.riv['text'] == idx].iloc[0]['geometry'].centroid
              toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                                    'longitude': geometry.x}
              
              toponyms_metadata[i][j]['type'] = str('riv')
            elif toponym_type == 'town':
              #Town
              toponyms_metadata[i][j]['coordinates'] = {'latitude': self.towns[toponym]['latitude'],
                                                      'longitude': self.towns[toponym]['longitude']}
              toponyms_metadata[i][j]['type'] = str('town')
              if len(found) == 3: #The found list is three-items long: this means there were more than one entry for a town!
                #TODO: We now do a very simple strategy: output all possible ambiguous names to the list and be off with it.
                #This will need to be changed in the future
                toponyms_metadata[i][j]['ALTERNATIVES'] = {'coordinates': [], 'type': []}

                if toponym != 'La Palma':
                  for repeated_entry in self.towns['REPEATED_TOWNS'][toponym]:
                    toponyms_metadata[i][j]['ALTERNATIVES']['coordinates'].append({'latitude': repeated_entry['latitude'],
                                                        'longitude': repeated_entry['longitude']})
                    toponyms_metadata[i][j]['ALTERNATIVES']['type'].append('town')
                else:
                  repeated_entry = self.towns['La Palma de Gran Canaria']
                  toponyms_metadata[i][j]['ALTERNATIVES']['coordinates'].append({'latitude': repeated_entry['latitude'],
                                                        'longitude': repeated_entry['longitude']})
                  toponyms_metadata[i][j]['ALTERNATIVES']['type'].append('town')
            
            elif toponym_type == 'prov':
              #Province
              if found[1] != '':
                if not simplifyPolylines:
                  toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.prov.iloc[found[1]]['geometry'])
                else:
                  geometry = self.prov.iloc[found[1]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                  
                geometry = self.prov.iloc[found[1]]['geometry'].centroid
                toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                                    'longitude': geometry.x}
                
              else:
                if not simplifyPolylines:
                  toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.prov.loc[self.alt_prov_names[toponym]]['geometry'])
                else:
                  geometry = self.prov.loc[self.alt_prov_names[toponym]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                geometry = self.prov.loc[self.alt_prov_names[toponym]]['geometry'].centroid
                toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                                    'longitude': geometry.x}
              toponyms_metadata[i][j]['type'] = str('prov')


          else:
            if found[0] == 'comm':
              if found[1] != '':
                if not simplifyPolylines:
                  toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.comm.iloc[found[1]]['geometry'])
                else:
                  geometry = self.comm.iloc[found[1]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                geometry = self.comm.iloc[found[1]]['geometry'].centroid
                toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                                    'longitude': geometry.x}
              else:
                if not simplifyPolylines:
                  toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.comm.loc[self.alt_comm_names[toponym]]['geometry'])
                else:
                  geometry = self.comm.loc[self.alt_comm_names[toponym]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                geometry = self.comm.loc[self.alt_comm_names[toponym]]['geometry'].centroid
                toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                                    'longitude': geometry.x}
              toponyms_metadata[i][j]['type'] = str('comm')
            elif found[0] == 'dam':
              if not simplifyPolylines:
                try:
                  toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.dams.iloc[found[1]]['geometry'])
                  toponyms_metadata[i][j]['type'] = str('dam')
                  geometry = self.dams.iloc[found[1]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                                    'longitude': geometry.x}
                except IndexError:
                  toponyms_metadata[i][j]['coordinates'] = None
                  toponyms_metadata[i][j]['type'] = 'UNK'
                  toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': 0,
                                                                    'longitude': 0}
              else:
                try:
                  geometry = self.dams.iloc[found[1]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                  toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                  toponyms_metadata[i][j]['type'] = str('dam')
                except IndexError:
                  toponyms_metadata[i][j]['coordinates'] = None
                  toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': 0,
                                                                    'longitude': 0}
                  toponyms_metadata[i][j]['type'] = 'UNK'
            
            elif found[0] == 'prov':
              if found[1] != '':
                if not simplifyPolylines:
                  toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.prov.iloc[found[1]]['geometry'])
                else:
                  geometry = self.prov.iloc[found[1]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                geometry = self.prov.iloc[found[1]]['geometry'].centroid
                toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
              else:
                if not simplifyPolylines:
                  toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.prov.loc[self.alt_prov_names[toponym]]['geometry'])
                else:
                  geometry = self.prov.loc[self.alt_prov_names[toponym]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                geometry = self.prov.loc[self.alt_prov_names[toponym]]['geometry'].centroid
                toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
              toponyms_metadata[i][j]['type'] = str('prov')
            elif found[0] == 'riv':
              if found[1] != '':
                if not simplifyPolylines:
                  toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.riv.iloc[found[1]]['geometry'])
                else:
                  geometry = self.riv.iloc[found[1]]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                geometry = self.riv.iloc[found[1]]['geometry'].centroid
                toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
              else:
                idx = toponym.upper().replace('Á','A').replace('É','E').replace('Í','I').replace('Ó','O').replace('Ú','U')
                if idx in self.riv['text'].to_list():
                  if not simplifyPolylines:
                    toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.riv.loc[self.riv['text'] == idx].iloc[0]['geometry'])
                  else:
                    geometry = self.riv.loc[self.riv['text'] == idx].iloc[0]['geometry'].centroid
                    toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                  geometry = self.riv.loc[self.riv['text'] == idx].iloc[0]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                elif 'RIO ' + idx.strip() in self.riv['text'].to_list():
                  if not simplifyPolylines:
                    toponyms_metadata[i][j]['coordinates'] = shapely.to_geojson(self.riv.loc[self.riv['text'] == 'RIO ' + idx.strip()].iloc[0]['geometry'])
                  else:
                    geometry = self.riv.loc[self.riv['text'] == 'RIO ' + idx.strip()].iloc[0]['geometry'].centroid
                    toponyms_metadata[i][j]['coordinates'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
                  geometry = self.riv.loc[self.riv['text'] == 'RIO ' + idx.strip()].iloc[0]['geometry'].centroid
                  toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': geometry.y,
                                                        'longitude': geometry.x}
              toponyms_metadata[i][j]['type'] = str('riv')
            
            elif found[0] == 'town':
              toponyms_metadata[i][j]['coordinates'] = {'latitude': self.towns[toponym]['latitude'],
                                                        'longitude': self.towns[toponym]['longitude']}
              toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': self.towns[toponym]['latitude'],
                                                        'longitude': self.towns[toponym]['longitude']}
              toponyms_metadata[i][j]['type'] = str('town')
            
            elif found[0] == 'country':
              #If there is a reference to a country, mark it as a special case and add it to our database anyway.
              #You can later decide, after running inference, if you want to keep this information or discard it,
              #but it will be provided to the end user anyway
              toponyms_metadata[i][j]['coordinates'] = {'latitude': self.countries[found[1]]['latitude'],
                                                        'longitude': self.countries[found[1]]['longitude']}
              toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': self.countries[found[1]]['latitude'],
                                                        'longitude': self.countries[found[1]]['longitude']}
              toponyms_metadata[i][j]['type'] = str('country')

        else:
          #DID NOT FIND TOPONYM IN IGN DATABASE

          #2nd and last chance: the IGN database is incomplete regarding very small town names,
          #that information however IS found within the Geonames database. Try and see if we
          #can find one UNAMBIGUOUS reference within the database. We will only do this check
          #for towns, not for anything else!
          if toponym in self.geonames.keys() and len(self.geonames[toponym]) <= 1 and self.geonames[toponym][0]['feature code'].startswith('PPL'):
            toponyms_metadata[i][j]['coordinates'] = {'latitude': self.geonames[toponym][0]['latitude'],
                                                        'longitude': self.geonames[toponym][0]['longitude']}
            toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': self.geonames[toponym][0]['latitude'],
                                                                'longitude': self.geonames[toponym][0]['longitude']}
            toponyms_metadata[i][j]['type'] = 'town'
          else:
            if toponym in self.geonames['ALTERNATIVE_NAMES'].keys() and len(self.geonames['ALTERNATIVE_NAMES'][toponym]) <= 1 \
              and len(self.geonames[self.geonames['ALTERNATIVE_NAMES'][toponym][0]]) == 1 and self.geonames[self.geonames['ALTERNATIVE_NAMES'][toponym][0]][0]['feature code'].startswith('PPL'):
                toponyms_metadata[i][j]['coordinates'] = {'latitude': self.geonames[self.geonames['ALTERNATIVE_NAMES'][toponym][0]][0]['latitude'],
                                                          'longitude': self.geonames[self.geonames['ALTERNATIVE_NAMES'][toponym][0]][0]['longitude']}
                toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': self.geonames[self.geonames['ALTERNATIVE_NAMES'][toponym][0]][0]['latitude'],
                                                          'longitude': self.geonames[self.geonames['ALTERNATIVE_NAMES'][toponym][0]][0]['longitude']}
                toponyms_metadata[i][j]['type'] = 'town'
            else:
              #Toponym NOT found, fill it with dummy values
              toponyms_metadata[i][j]['coordinates'] = None
              toponyms_metadata[i][j]['coordinates_centroid_values'] = {'latitude': 0,
                                                                  'longitude': 0}
              toponyms_metadata[i][j]['type'] = 'UNK'
  
    return toponyms_metadata
  
  ##############################
  ## TOPONYMS DISAMBIGUATION ##
  #############################

  def disambiguate_toponym_type(self,toponym,doc):
    #Function that attempts to disambiguate the toponym type of names that are shared across rivers, towns and provinces
    #(Río Turia vs. Turia; Huesca vs. provincia de Huesca). It does so via the use of some heuristic rules based on
    #the syntactic analysis of the context surrounding the detected toponym names. The syntactic analysis is performed via spaCy,
    #a library that had been used in a prior step to split each text into the set of its sentences. Other than sentence
    #splitting, we also keep the dependency relations of each of the tokens, as output by spaCy, and use them here to
    #check the syntactic relationships between a toponym and its surrounding words.

    token_type = 'town' #Return this value by default (default to "town")
    for token in doc:
          if token.text == toponym:

            #1a) "río + Name"
            if token.head.text == 'río' or token.head.text == 'cuenca' or token.head.text == 'confederación' or token.head.text == 'confederación' or token.head.text == 'ribera' and token.dep_ == 'appos':
              token_type = 'riv'  #stream
            #1b) "provincia de + Name"
            elif token.head.text == 'provincia' and token.dep_ == 'nmod':
              token_type = 'prov'
            elif token.head.text == 'presa' or token.head.text == 'embalse' and token.dep_ == 'nmod':
              token_type = 'dam'

            #2a) "ríos/cuencas/provincias de X, Y, Name and Z"
            elif token.head.tag_ == 'PROPN':
              head = token.head
              if token.dep_ == 'appos':
                if head.head.text == 'ríos' or head.head.text == 'cuencas' or head.head.text == 'confederaciones' or head.head.text == 'riberas':
                  token_type = 'riv'  #stream
                elif head.head.text == 'provincias':
                  token_type = 'prov'
                elif head.head.text == 'presas' or head.head.text == 'embalses':
                  token_type = 'dam'

            #2b) "ríos/cuencas/provincias de Name, Y and Z"
            elif token.head.tag_ == 'NOUN' and token.dep_ == 'nmod' or token.dep_ == 'conj':
              if token.head.text == 'ríos' or token.head.text == 'cuencas' or token.head.text == 'confederaciones' or head.head.text == 'riberas':
                token_type = 'riv'
              elif token.head.text == 'provincias':
                token_type = 'prov'
              elif token.head.text == 'presas' or token.head.text == 'embalses':
                token_type = 'dam'
            
            else:
              #3- Det. + "river name" ("el Turia"/"al Turia"/"del Turia") >> REASON: In case of doubt, in Spanish, usually when an ambiguous name that
              #can refer to both a city and a river, it is usually a river (or a sports team; "el Barcelona") when precedeed by an article ("el Turia").
              for left_token in token.lefts:
                #3a) "el Turia"
                if left_token.dep_ == 'det' and left_token.head.text == token.text:
                  if left_token.text == 'el':
                    token_type = 'riv'
                    break
                #3b) "al/del Turia"
                elif left_token.dep_ == 'case' and left_token.head.text == token.text:
                  if left_token.text == 'al' or left_token.text == 'del':
                    token_type = 'riv'
                    break

    return token_type
  
  def geolocation(self,toponyms,toponyms_metadata,doc,doc_sentence,simplifyPolylines=False):
    return self.geolocation_IGN(toponyms,toponyms_metadata,doc,doc_sentence,simplifyPolylines)
  
  #########################
  ## CLASS' CALL METHOD ##
  ########################

  def __call__(self,text,doc,doc_sentence):
    
    toponyms = []
    toponyms_metadata = []

    #Call NER model to predict tokens
    predicted_token_class = self.pipe(text)

    #Do token aggregation over the output of the Transformer-based model
    toponyms, toponyms_metadata = self.loc_tokens_aggregation(predicted_token_class,text)

    #Retrieve geolocation of located toponyms and output coordinates for each of the found toponyms
    if self.do_geocoding:
      toponyms_metadata = self.geolocation(toponyms,toponyms_metadata,doc,doc_sentence)
    
    return toponyms,toponyms_metadata
  
