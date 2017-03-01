#!/bin/bash

fout="model_output.txt"
delim="########################################################################"

# Get Dataset
python Notebook_mining.py > $fout 2>&1

# Hyperparamter Tunning
echo -e "\n\n\n $delim \n\t\t\tGRIDSEARCH\n $delim \n\n\n " >> $fout
python Notebook_gs_logit.py >> $fout 2>&1

echo -e "\n\n\n $delim \n\t\t\tGRIDSEARCH\n $delim \n\n\n " >> $fout
python Notebook_gs_svc.py >> $fout 2>&1

echo -e "\n\n\n $delim \n\t\t\tGRIDSEARCH\n $delim \n\n\n " >> $fout
python Notebook_gs_nb.py >> $fout 2>&1

echo -e "\n\n\n $delim \n\t\t\tGRIDSEARCH\n $delim \n\n\n " >> $fout
python Notebook_gs_sgd.py >> $fout 2>&1

