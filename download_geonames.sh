#Download and unzip Geonames Spain dumped local file

mkdir ./seqia/geonames/
wget https://download.geonames.org/export/dump/ES.zip
mv ./ES.zip ./seqia/geonames/
mkdir ./seqia/geonames/ES/
unzip ./seqia/geonames/ES.zip -d ./seqia/geonames/ES/
rm ./seqia/geonames/ES.zip
