#!/bin/bash

date > GRIDSEARCH_OUTPUT.txt
python Notebook.py >> GRIDSEARCH_OUTPUT.txt 2>&1
date >> GRIDSEARCH_OUTPUT.txt

