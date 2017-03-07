from Notebook_helperfunctions import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline                                           
from sklearn.feature_extraction.text import TfidfVectorizer                     
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

### INPUT
args = gs_cml()
f_authorship = args.infile

### PREPROCESSING
df = pd.read_csv(f_authorship)
df.drop_duplicates()

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
#df['text'] = df['text'].apply(preprocessor)

X = df.loc[:, 'text'].values
y = df.loc[:, 'user_id'].values
le = LabelEncoder()
y = le.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=1)
### Grid Search CV
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=1))])

ngram_range = [(1, args.ngram)]
param_range = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'vect__ngram_range': ngram_range,
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, 
                                   tokenizer_porter, 
                                   tokenizer_twitter],
               'vect__max_df': [1.0, 0.9, 0.8],
               'vect__norm':[None, 'l1', 'l2'],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': param_range},
              {'vect__ngram_range': ngram_range,
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, 
                                   tokenizer_porter, 
                                   tokenizer_twitter],
               'vect__max_df': [1.0, 0.9, 0.8],
               'vect__use_idf':[False],
               'vect__norm':[None, 'l1', 'l2'],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': param_range},
              ]

gs_lr_tfidf = GridSearchCV(lr_tfidf, 
                           param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=args.jobs)

lg_time0 = time.time()
gs_lr_tfidf.fit(X_train, y_train)
clf_lr = gs_lr_tfidf.best_estimator_
lg_time1 = time.time()
print('EXECUTION TIME for logit gs : {} secs\n\n'.format(lg_time1 - lg_time0))
print('Best parameter set: {} \n'.format(gs_lr_tfidf.best_params_))
print('CV Accuracy: {:.3f}'.format(gs_lr_tfidf.best_score_))
print('Test Accuracy: {:.3f}'.format(clf_lr.score(X_test, y_test)))

# Store model
dest = os.path.join('pkl_objects')
if os.path.exists(dest):
    pickle.dump(gs_lr_tfidf, open(
        os.path.join(dest, str(args.ngram)+'gs_logit.pkl'), 'wb'), protocol=4
    )
