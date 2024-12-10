# cbam_gvc

This package is used for the manuscript on expanding CBAM.


## Description

The analysis is carried out in cbam_gvc.py

This python script assumes that EXIOBASE has been first parsed to pickle once, to greatly increase execution speed.

To parse the text version of EXIOBASE to pickle:

- Download IOT_2019_pxp.zip from https://zenodo.org/records/5589597
- Unpack zip file to input/IOT_2019_pxp (i.e. the text files will now be stored as input/IOT_2019_pxp/A.txt, input/IOT_2019_pxp/x.txt, etc.)
- Run parse_eb.py

parse_eb will parse the text version of EXIOBASE to pickle in the input folder, as well as calculate and store the Leontief inverse.

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.1.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
