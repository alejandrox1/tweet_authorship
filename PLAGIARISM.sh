#!/bin/bash

# output options                                                                
fbase="model_output.txt"                                                        
fout=${maxgram}gram_${fbase}                                                    
delim="#######################################################################"

# Get command line arguments (if provided)
while [[ $# > 1 ]]; do
        key="$1"
	case $key in
		--batch)
		sbatch="$2"
		shift
		;;
		--tweet_lim)
		tweet_lim="$2"
		shift
		;;
        	--ngram)
            	maxgram="$2"
            	shift 
            	;;
            	--jobs)
            	njobs="$2"
            	shift
            	;;
            	*)
		echo "Unkown option."
                exit
         	;;
        esac
        shift 
done

# output 
echo "Running Gridsearches with max ngram: \
	$maxgram and $njobs CPUs" > $fout 2>&1

# Get Dataset
python Notebook_mining.py \
	--batch $sbatch --tweet_lim $tweet_lim >> $fout 2>&1

# Hyperparameter Tuning
echo -e "\n\n\n $delim \n\t\t\tGRIDSEARCH\n $delim \n\n\n " >> $fout
python Notebook_gs_logit.py \
	--ngram ${maxgram:-1} --jobs ${njobs:--1} >> $fout 2>&1

echo -e "\n\n\n $delim \n\t\t\tGRIDSEARCH\n $delim \n\n\n " >> $fout
python Notebook_gs_svc.py \
	--ngram ${maxgram:-1} --jobs ${njobs:--1} >> $fout 2>&1

echo -e "\n\n\n $delim \n\t\t\tGRIDSEARCH\n $delim \n\n\n " >> $fout
python Notebook_gs_nb.py \
	--ngram ${maxgram:-1} --jobs ${njobs:--1} >> $fout 2>&1

