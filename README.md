# Emotone - Melody generation based on emotion
## HCAI project

### Author:
- Joana Simoes, No. 2019217013

### Objective
This project aims to create an artificial intelligence system capable of generating melodies that induce certain emotions in people.

### Important Notes:
- Due to the large size of the LSTM checkpoints, before running the project (in case you don't want to retrain the LSTM) you need to download the zip files from https://drive.google.com/file/d/1q6z1J7TDpdTiqzRhH4fudDKtcuy_g5wV/view?usp=sharing and add the checkpoints to the project's "trained" folder

- The project was elaborated in Google Colab, being recommended to use it to run the project. The python version used was 3.8.16, so there may be incompatibilities with other versions.

- The files that come with the project contain the project code, this notebook "project.ipynb" only serves to "put the pieces together in one place",  and it is advised to only run the project from the notebook.

- The sections should be run sequentially, in case you want to run the whole program (train the system from scratch). Note that this option may take more than 15 hours, depending on the machines. 

- To save time, checkpoints have been created throughout the notebook, so that each part can be run separately. 

- The sections 'Imports', 'Variables', should always be run first. 

- To train the classifier, change the weights of the LSTM (Evolve LSTM) or generate tunes, the "Variable post-training LSTM" must be run before them.


### References

The project was heavily based on the work of Lucas N. Ferreira and Jim Whitehead, their work can be accessed at https://arxiv.org/abs/2103.06125.
