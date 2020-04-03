# README

## Organization of this project

This project has 3 directories: B for task B, D for task D
and Datasets for data.

Datasets directory contains training raw datasets in Subtasks_BD and test raw dataset SemEval2017-task4-test.subtask-BD.english.txt.
Datasets directory also contains preprocessing.py which is used to pre-process raw data and generate training data and test data for this project.
The new training data and text data are written into 2 .csv files separately.

B directory and D directory have the same organization. 
They both have a .py file to construct and train a model, but models have different architecture.
After training, the model is saved in .h5 file.
There are also 3 .png files to visualize training process.

This project also has 3 .py files: tokenizer.py, main.py and further_test.py. 

tokenizer.py is used to train a tokenizer for word representation. 
The tokenizer was saved in Tok.pickle. 

main.py is used to test trained models which have already been saved in B directory and D directory.
It also writes predictions of test data into B.csv and D.csv respectively.

further_test.py is used to do further tests using scikit-learn based on predictions which have been written into B.csv and D.csv.

## Packages required

numpy, pandas, matplotlib, tensorflow1.15, keras, scikit-learn