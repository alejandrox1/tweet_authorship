from Notebook_helperfunctions import *
from sklearn.linear_model import SGDClassifier

### INPUT
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

sgd_tfidf = Pipeline([('vect', tfidf),
                     ('clf', SGDClassifier(random_state=42)),])

ngram_range = [(1, 1)]
param_range = [0.25, 0.5, 0.75, 1.0]
param_grid = [{'vect__ngram_range': ngram_range,
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'clf__loss' : ['hinge', 'log'],
               'clf__penalty' : ['l1', 'l2'],
               'clf__n_iter' : [3,5,7],
               'clf__alpha': param_range},
              {'vect__ngram_range': ngram_range,
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__loss' : ['hinge', 'log'],
               'clf__penalty' : ['l1', 'l2'],
               'clf__n_iter' : [3,5,7],
               'clf__alpha': param_range}]

gs_sgd_tfidf = GridSearchCV(estimator=sgd_tfidf, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=5,
                  verbose=1,
                  n_jobs=-1)

sgd_time0 = time.time()
gs_sgd_tfidf.fit(X_train, y_train)
sgd_time1 = time.time()
print('EXECUTION TIME for sgd gs : {} secs\n\n'.format(sgd_time1 - sgd_time0))
print('Best parameter set: {} \n'.format(gs_sgd_tfidf.best_params_))
print('CV Accuracy: {:.3f}'.format(gs_sgd_tfidf.best_score_))
clf_sgd = gs_sgd_tfidf.best_estimator_
print('Test Accuracy: {:.3f}'.format(clf_sgd.score(X_test, y_test)))

# Store model
dest = os.path.join('pkl_objects')
if os.path.exists(dest):
    pickle.dump(gs_sgd_tfidf, open(os.path.join(dest, 'gs_sgd.pkl'), 'wb'),
                protocol=4)
