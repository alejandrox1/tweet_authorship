# Plagiarism Detection in Twitter
Determining tweet authorship.

# Workflow
[PLAGIARISM.sh](https://github.com/alejandrox1/tweet_authorship) can be used to run a series of steps to obtain the tweets of different users, clean the data, and store is in a `csv` format for later use. This script will also perform various `GridSearches` in order to find the best parameters for machine learning classifiers that will later be used to tell whether a tweet belogs or not to a given account.

[plagiarism.ipynb](https://github.com/alejandrox1/tweet_authorship) contains the validation of ensemble learners and learning curves to visualize the quality of our models.

# Model Details
* [setup_t2micro.sh](https://github.com/alejandrox1/tweet_authorship/blob/master/setup_t2micro.sh) <br/>
 Set up an `AWS EC2` instance to perform calculations.

* [plagiarism.sh](https://github.com/alejandrox1/tweet_authorship/blob/master/PLAGIARISM.sh) <br/>
 Script to set up the environment. The set up includes obtaining and cleaning tweets from accounts specified in
 [Notebook_mining.py](https://github.com/alejandrox1/tweet_authorship/blob/master/Notebook_gs_sgd.py) and hyperparameter tunning for different classifiers via the `Notebook_gs_*` scripts.

* [Notebook_helperfunctions.py](https://github.com/alejandrox1/tweet_authorship/blob/master/Notebook_helperfunctions.py) <br/>
  Functions to for data mining and preprocessing steps.
  
* [Notebook_mining.py](https://github.com/alejandrox1/tweet_authorship/blob/master/Notebook_mining.py) <br/>
  Obtaining and cleaning up a twitter data set for plagiarism detection.

* [Notebook_gs_logit.py](https://github.com/alejandrox1/tweet_authorship/blob/master/Notebook_gs_logit.py) <br/>
  Parameter tunning for `Logistic regression` model.

* [Notebook_gs_svc.py](https://github.com/alejandrox1/tweet_authorship/blob/master/Notebook_gs_svc.py) <br/>
  Parameter tunning for `Support Vector Machine` model.
  
* [Notebook_gs_nb.py](https://github.com/alejandrox1/tweet_authorship/blob/master/Notebook_gs_nb.py) <br/>
  Parameter tunning for `Naive Bayes` model.
