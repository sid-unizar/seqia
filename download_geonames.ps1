
#Windows-based PowerShell script to download the Geonames Spanish database dump

Invoke-WebRequest -uri "https://download.geonames.org/export/dump/ES.zip" -OutFile "$PSScriptRoot\seqia\geonames\ES.zip"

New-Item -ItemType Directory -Force -Path "$PSScriptRoot\seqia\geonames\ES"

Expand-Archive "$PSScriptRoot\seqia\geonames\ES.zip" -DestinationPath "$PSScriptRoot\seqia\geonames\ES"

Remove-Item "$PSScriptRoot\seqia\geonames\ES.zip"
