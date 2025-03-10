#Script that downloads all model weight's from the Github repository.
#You will need to install the GitHub CLI commands
#and setup the login credentials accordingly.
#If you can't run this script, simply download the files manually from the Github page and transfer them to your machine.

declare -a arr=("binary binary_model" "Agricultura impacts/Agricultura" "Ganaderia impacts/Ganadería" "Recursos.hidricos impacts/Recursos_hídricos" "Energetico impacts/Energético")

for i in "${arr[@]}"
do
   set -- $i
   gh release download --repo sid-unizar/seqia -p $1.bin -D ./seqia/models/$2/
   rename ./seqia/models/$2/$1.bin ./seqia/models/$2/pytorch_model.bin
done
