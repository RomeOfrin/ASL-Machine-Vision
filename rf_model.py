import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


model = RandomForestClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)


best_model = grid_search.best_estimator_


y_predict = best_model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('Best parameters found by grid search:')
print(grid_search.best_params_)
print('\nAccuracy: {:.2f}%'.format(score * 100))


f = open('model.p', 'wb')
pickle.dump({'model': best_model}, f)
f.close()
