from Notebook_helperfunctions import *
from sklearn.svm import SVC

### INPUT
args = cml()
f_authorship = 'users/authorship.csv'

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=1)
### Grid Search CV
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

svc_tfidf = Pipeline([('vect', tfidf),
                     ('clf', SVC(random_state=1))])

ngram_range = [(1, args.ngram)]
param_range = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'vect__ngram_range': ngram_range,
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'clf__C': param_range, 
               'clf__kernel': ['linear']},
              {'vect__ngram_range': ngram_range,
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__C': param_range, 
               'clf__kernel': ['linear']},
              {'vect__ngram_range': ngram_range,
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'clf__C': param_range, 
               'clf__gamma': param_range, 
               'clf__kernel': ['rbf']},
              {'vect__ngram_range': ngram_range,
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__C': param_range, 
               'clf__gamma': param_range, 
               'clf__kernel': ['rbf']}]

gs_svc_tfidf = GridSearchCV(estimator=svc_tfidf, 
                            param_grid=param_grid, 
                            scoring='accuracy', 
                            cv=5,
                            verbose=1,
                            n_jobs=args.jobs)

svc_time0 = time.time()
gs_svc_tfidf.fit(X_train, y_train)
svc_time1 = time.time()
print('EXECUTION TIME for svc gs : {} secs\n\n'.format(svc_time1 - svc_time0))
print('Best parameter set: {} \n'.format(gs_svc_tfidf.best_params_))
print('CV Accuracy: {:.3f}'.format(gs_svc_tfidf.best_score_))
clf_svc = gs_svc_tfidf.best_estimator_
print('Test Accuracy: {:.3f}'.format(clf_svc.score(X_test, y_test)))

# Store model
dest = os.path.join('pkl_objects')
if os.path.exists(dest):
    pickle.dump(gs_svc_tfidf, open(os.path.join(dest, str(args.ngram)+'gs_svc.pkl'), 'wb'),
                protocol=4)
