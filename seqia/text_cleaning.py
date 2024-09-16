
#Article cleaning function
def clean_text(text):

  import re

  #Cleans up some garbage HTML tags from news body text
  text = text.replace('&quot;','\'')
  text = text.replace(u'\xa0', u' ')

  #Get all different quote styles and unify them under a unique one
  text = text.replace('“',"\"")
  text = text.replace("”", "\"")
  text = text.replace("«", "\"")
  text = text.replace("»", "\"")
  text = text.replace("\'", "\"")

  #Match and remove HTML tags like this one: &#039;
  text = re.sub(r'&#[0-9]+;', '', text)

  #Clean multiple spaces and output them as just one
  text = re.sub("\s\s+", " ", text)

  #TODO: Add more cleaning as you see more details to be cleaned

  return text