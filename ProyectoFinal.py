import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
from sklearn.model_selection import validation_curve
from sklearn.utils import resample
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

Messias = pd.read_csv("df_messi_DTC.csv")
Messias ["Date"] = Messias["Date"].str.replace("-", "/")


time_data = [f.split('/') for f in Messias['Date']]
time_data_df = pd.DataFrame(time_data)
time_data_df.columns = ['Mes', 'Día', 'Año'] 


Messias['Día'] = time_data_df['Día'].astype(int)
Messias['Mes'] = time_data_df['Mes'].astype(int)
Messias['Año'] = time_data_df['Año'].astype(int)

Messias['Type'].value_counts()
penal_tlibre = []
en_juego = []
for valor in Messias['Type']:
      if valor != 'Penalty' and valor != 'Direct free kick':
             penal_tlibre.append(False)
             en_juego.append(True)
      else:
             penal_tlibre.append(True)
             en_juego.append(False)

Messias['penal_tlibre'] = pd.DataFrame(penal_tlibre)
Messias['en_juego'] = pd.DataFrame(en_juego)


Messias["Result"] = Messias["Result"].str.replace(" AET", "")

result_data = [f.split(':') for f in Messias['Result']]

for i, val in enumerate(result_data):
        if Messias['Venue'][i] == 'A':
            val.reverse()
            
result_data_df = pd.DataFrame(result_data)
result_data_df.columns = ['Result_fav', 'Result_contra']
Messias['Result_fav'] = result_data_df['Result_fav'].astype(int)
Messias['Result_contra'] = result_data_df['Result_contra'].astype(int)

Messias['victoria'] =  Messias['Result_fav'] > Messias['Result_contra']


Messias['Playing_Position'].value_counts()

Messias['Playing_Position'] = Messias["Playing_Position"].str.replace(" ", "")
Messias['Playing_Position'] = Messias["Playing_Position"].str.replace("AM", "SS")
Messias['Playing_Position'] = Messias["Playing_Position"].str.replace("LW", "SS")
Messias['Playing_Position'].value_counts()


result_gol_data = [f.split(':') for f in Messias['At_score']]
for indx, val2 in enumerate(result_gol_data):
    if Messias['Venue'][indx] == 'A':
        val2.reverse()

result_gol_data_df = pd.DataFrame(result_gol_data)

result_gol_data_df.columns = ['Result_fav_gol', 'Result_contra_gol']
Messias['Result_fav_gol'] = result_gol_data_df['Result_fav_gol'].astype(int)
Messias['Result_contra_gol'] = result_gol_data_df['Result_contra_gol'].astype(int)    

print(Messias)

messi_df_4encod= Messias.drop(['Season', 'Competition', 'Matchday', 'Date', 'Club',
    'Opponent', 'Result', 'Minute', 'At_score', 'Type',
    'Goal_assist', 'Minute_decenas', 'primer_tiempo', 'segundo_tiempo',
    'Día', 'Mes', 'Año', 'penal_tlibre', 'en_juego', 'Result_fav',
    'Result_contra', 'victoria', 'Result_fav_gol', 'Result_contra_gol'], axis=1)
encodedVenue = pd.get_dummies(messi_df_4encod.Venue, prefix= 'Venue')
encodedPosition = pd.get_dummies(messi_df_4encod.Playing_Position, prefix= 'Playing_Position')
categorias_eval = pd.concat([encodedVenue,encodedPosition], axis=1,ignore_index=False)

Messias = pd.concat([Messias, categorias_eval], axis=1,ignore_index=False)

Messias['penal_tlibre'] = Messias['penal_tlibre'].astype(int)
Messias['en_juego'] = Messias['en_juego'].astype(int)
Messias['primer_tiempo'] = Messias['primer_tiempo'].astype(int)
Messias['segundo_tiempo'] = Messias['segundo_tiempo'].astype(int)


Messias = Messias.drop(Messias[Messias['primer_tiempo'] == False].index)

Messias = Messias.drop(columns = ['Competition', 'Season', 'Matchday', 'Date', 'Goal_assist', 'primer_tiempo', 
    'segundo_tiempo', 'Result', 'Type', 'Result_fav', 'Result_contra', 'Club', 'Opponent', 'At_score', 'Venue' , 'Playing_Position', 'Año'])

# df_majority = Messias[Messias['victoria'] == 1]
# df_minority = Messias[Messias['victoria'] == 0]

# n_samples_to_keep = 200  # Número deseado de instancias con 'target' igual a 0
# df_majority_subsampled = resample(df_majority, replace=False, n_samples=n_samples_to_keep, random_state=42)

# Messias = pd.concat([df_majority_subsampled, df_minority])

print(Messias['victoria'].value_counts())


      
################################################################################################################################
# Termina limpieza de datos y empieza modelos
################################################################################################################################
# Empieza la separación en Train y Test y se realiza un Random Search

x = Messias.drop('victoria', axis = 1)
y = Messias['victoria']

x_train, x_test ,y_train, y_test = train_test_split( x , y, random_state=10)

################################################################################################################################
# Termina la separación en Train y Test
################################################################################################################################
# Función para plotear línea de Aprendizaje 

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.01, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Puntuación")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Puntuación de entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Puntuación de validación cruzada")

    plt.legend(loc="best")
    return plt

################################################################################################################################
# Termina función para plotear línea de Aprendizaje 
################################################################################################################################
################################################################################################################################
# RANDOM SEARCH
################################################################################################################################
################################################################################################################################
# Decision Tree
    
# param_dist = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': np.arange(1, 21),
#     'min_samples_split': np.arange(2, 11),
#     'min_samples_leaf': np.arange(1, 11),
#     'max_features': ['auto', 'sqrt', 'log2', None]
# }
 
# tree = DecisionTreeClassifier()

# random_search = RandomizedSearchCV(tree, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)

# random_search.fit(x_train, y_train)

# best_params = random_search.best_params_

# treebest = DecisionTreeClassifier(**best_params)

# treebest.fit(x_train, y_train)

# scores = cross_val_score(treebest, x_train, y_train, cv=5) 
# print("Puntuaciones de validación cruzada:", scores)
# print("Puntuación media:", scores.mean())

# accuracy = treebest.score(x_test, y_test)
# y_pred = treebest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision = ' , precision_score(y_test,y_pred,average='macro'))


# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=treebest.classes_)
# disp.plot() 

# title = "Decision Tree"

# plt.title(title)
# plt.show()

# plot_learning_curve(treebest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina Decision Tree
################################################################################################################################
# Random Forest 

# param_dist = {
#     'n_estimators': [50, 100, 150, 200],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2', None],
#     'bootstrap': [True, False]
# }

# rfc = RandomForestClassifier()

# random_search = RandomizedSearchCV(rfc, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)

# random_search.fit(x_train, y_train)

# best_params = random_search.best_params_

# rfcbest = RandomForestClassifier(**best_params)

# rfcbest.fit(x_train, y_train)

# scores = cross_val_score(rfcbest, x_train, y_train, cv=5) 
# print("Puntuaciones de validación cruzada:", scores)
# print("Puntuación media:", scores.mean())

# accuracy = rfcbest.score(x_test, y_test)
# y_pred = rfcbest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision= ' , precision_score(y_test,y_pred,average='macro'))

# print()

# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=rfcbest.classes_)
# disp.plot()

# title = "Random Forest"

# plt.show()

# plot_learning_curve(rfcbest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina Random Forest
################################################################################################################################
# Histogram-Based Gradient Boosting

# param_dist = {
#     'learning_rate': [0.01, 0.1, 0.2, 0.3],
#     'max_iter': [50, 100, 150, 200],
#     'max_leaf_nodes': [15, 31, 63, None],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_leaf': [1, 2, 4],
#     'l2_regularization': [0.0, 0.1, 0.2, 0.3]
# }

# hgbc = HistGradientBoostingClassifier()

# random_search = RandomizedSearchCV(hgbc, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)

# random_search.fit(x_train, y_train)

# best_params = random_search.best_params_

# hgbcbest = HistGradientBoostingClassifier(**best_params)

# hgbcbest.fit(x_train, y_train)

# scores = cross_val_score(hgbcbest, x_train, y_train, cv=5) 
# print("Puntuaciones de validación cruzada:", scores)
# print("Puntuación media:", scores.mean())

# accuracy = hgbcbest.score(x_test, y_test)
# y_pred = hgbcbest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision= ' , precision_score(y_test,y_pred,average='macro'))

# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=hgbcbest.classes_)
# disp.plot() 
# title = 'Histogram-Based Gradient Boosting'

# plt.show()


# plot_learning_curve(hgbcbest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina Histogram-Based Gradient Boosting
################################################################################################################################
# AdaBoost

param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [1],
    'algorithm': ['SAMME', 'SAMME.R']
}

adab = AdaBoostClassifier()
class_weights = {0: 0.8, 1: 0.2} 

random_search = RandomizedSearchCV(adab, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=10)

random_search.fit(x_train, y_train, sample_weight=[class_weights[y] for y in y_train])



best_params = random_search.best_params_


print(best_params)

adabest = AdaBoostClassifier(**best_params)

adabest.fit(x_train, y_train)
param_range = np.arange(1, 11)
train_scores, test_scores = validation_curve(
    adabest, x_train, y_train, param_name="n_estimators", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=-1)
scores = cross_val_score(adabest, x_train, y_train, cv=5) 
print("Puntuaciones de validación cruzada:", scores)
print("Puntuación media:", scores.mean())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
msle_scores = cross_val_score(adabest, x_train, y_train, cv=kf, scoring='neg_mean_squared_log_error')
msle_scores = -msle_scores


accuracy = adabest.score(x_test, y_test)
y_pred = adabest.predict(x_test)
f1 = f1_score(y_test, y_pred)


print("F1-score:", f1)
print('accuracy =' , accuracy_score(y_test,y_pred))
print('precision= ' , precision_score(y_test,y_pred,average='micro', zero_division=0))
print("Mean Squared Logarithmic Error:", (np.mean(scores)))

matriz=confusion_matrix(y_test,y_pred)
plt.rcParams["figure.dpi"] = 200
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=adabest.classes_)
disp.plot() 
title = 'AdaBoost'
plt.show()


TN, FP, FN, TP = matriz.ravel()
FPR = FP / (FP + TN)
print("Tasa de Falsos Positivos (FPR):", FPR)

plot_learning_curve(adabest, title, x_train, y_train, cv=4, n_jobs=-1)
plt.show()



train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, color="blue", marker="o", markersize=5, label="Training Accuracy")
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")
plt.plot(param_range, test_mean, color="green", linestyle="--", marker="s", markersize=5, label="Validation Accuracy")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color="green")

plt.title("Curva de Validación para AdaBoost")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.xticks(param_range)
plt.legend(loc="lower right")
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(range(len(msle_scores)), msle_scores)
plt.xlabel('Fold de Validación')
plt.ylabel('MSLE')
plt.title(f'MSLE con Validación Cruzada (Promedio MSLE: {np.mean(msle_scores):.4f})')
plt.show()



estimator_range = range(51, 201, 15)

train_errors_mean = 1 - np.mean(train_scores, axis=1)
test_errors_mean = 1 - np.mean(test_scores, axis=1)
train_errors_std = np.std(train_scores, axis=1)
test_errors_std = np.std(test_scores, axis=1)

# Plotea la curva de sesgo-contra-varianza
plt.figure(figsize=(10, 6))
plt.plot(estimator_range, train_errors_mean, label='Sesgo (entrenamiento)')
plt.plot(estimator_range, test_errors_mean, label='Error de Generalización (prueba)')
plt.fill_between(estimator_range, train_errors_mean - train_errors_std,
                 train_errors_mean + train_errors_std, alpha=0.2)
plt.fill_between(estimator_range, test_errors_mean - test_errors_std,
                 test_errors_mean + test_errors_std, alpha=0.2)
plt.xlabel('Número de Estimadores (n_estimators)')
plt.ylabel('Error')
plt.title('Curva de Sesgo-Contra-Varianza para AdaBoost')
plt.legend()
plt.grid(True)
plt.show()


bias_squared_list = []
variance_list = []
base_model = DecisionTreeClassifier(max_depth=1)
for n_estimators in estimator_range:
    model = AdaBoostClassifier(base_model, n_estimators=n_estimators)
    
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)   
    
    bias_squared = (1 - train_accuracy) ** 2
    variance = (1 - test_accuracy) ** 2 
   
    bias_squared_list.append(bias_squared)
    variance_list.append(variance)

plt.figure(figsize=(10, 6))
plt.plot(estimator_range, bias_squared_list, label='Sesgo al Cuadrado')
plt.plot(estimator_range, variance_list, label='Varianza')
plt.xlabel('Número de Estimadores (n_estimators)')
plt.ylabel('Valor')
plt.title('Sesgo al Cuadrado vs. Varianza para AdaBoost')
plt.legend()
plt.grid(True)
plt.show()







################################################################################################################################
# Termina AdaBoost
################################################################################################################################
#Multi-layer Perceptron 

# param_dist = {
#     'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128)],
#     'activation': ['relu', 'tanh', 'logistic'],
#     'solver': ['adam', 'sgd'],
#     'alpha': uniform(0.0001, 0.01),
#     'learning_rate': ['constant', 'invscaling', 'adaptive'],
#     'max_iter': randint(100, 1000)
# }

# mlp = MLPClassifier()

# random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)

# random_search.fit(x_train, y_train)

# best_params = random_search.best_params_

# mlpbest = MLPClassifier(**best_params)

# mlpbest.fit(x_train, y_train)

# scores = cross_val_score(mlpbest, x_train, y_train, cv=5) 
# print("Puntuaciones de validación cruzada:", scores)
# print("Puntuación media:", scores.mean())

# accuracy = mlpbest.score(x_test, y_test)
# y_pred = mlpbest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision= ' , precision_score(y_test,y_pred,average='macro'))

# print()

# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=mlpbest.classes_)
# disp.plot() 
# title = 'Multi-layer Perceptron'
# plt.show()

# plot_learning_curve(mlpbest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina Multi-layer Perceptron
################################################################################################################################
################################################################################################################################
# GRID SEARCH
################################################################################################################################
################################################################################################################################
# Decision Tree
    
# param_grid = {
#     'criterion': ['gini', 'entropy'], 
#     'max_depth': [None, 10, 20, 30],  
#     'min_samples_split': [2, 5, 10],  
#     'min_samples_leaf': [1, 2, 4]  
# }
 
# tree = DecisionTreeClassifier()

# grid_search = GridSearchCV(tree, param_grid=param_grid, cv=5, n_jobs=-1)

# grid_search.fit(x_train, y_train)

# best_params = grid_search.best_params_

# treebest = DecisionTreeClassifier(**best_params)

# treebest.fit(x_train, y_train)

# scores = cross_val_score(treebest, x_train, y_train, cv=5) 
# print("Puntuaciones de validación cruzada:", scores)
# print("Puntuación media:", scores.mean())

# accuracy = treebest.score(x_test, y_test)
# y_pred = treebest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision= ' , precision_score(y_test,y_pred,average='macro'))

# print()

# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=treebest.classes_)
# disp.plot() 
# title = 'Decision Tree'
# plt.show()

# plot_learning_curve(treebest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina Decision Tree
################################################################################################################################
# Random Forest 

# param_grid = {
#     'n_estimators': [50, 100, 200], 
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]  
# }

# rfc = RandomForestClassifier()

# grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5, n_jobs=-1)

# grid_search.fit(x_train, y_train)

# best_params = grid_search.best_params_

# rfcbest = RandomForestClassifier(**best_params)

# rfcbest.fit(x_train, y_train)

# scores = cross_val_score(rfcbest, x_train, y_train, cv=5) 
# print("Puntuaciones de validación cruzada:", scores)
# print("Puntuación media:", scores.mean())

# accuracy = rfcbest.score(x_test, y_test)
# y_pred = rfcbest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision= ' , precision_score(y_test,y_pred,average='macro'))

# print()

# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=rfcbest.classes_)
# disp.plot() 
# title = 'Random Forest'
# plt.show()

# plot_learning_curve(rfcbest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina Random Forest
################################################################################################################################
# Histogram-Based Gradient Boosting

# param_grid = {
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_iter': [100, 200, 300],
#     'max_depth': [3, 4, 5],
#     'min_samples_leaf': [1, 2, 4],
#     'max_leaf_nodes': [31, 63, 127],
#     'l2_regularization': [0.0, 0.1, 0.2],
#     'max_bins': [255, 511, 1023],
#     'validation_fraction': [0.1, 0.2, 0.3]
# }

# hgbc = HistGradientBoostingClassifier()

# grid_search = GridSearchCV(hgbc, param_grid=param_grid, cv=5, n_jobs=-1)

# grid_search.fit(x_train, y_train)

# best_params = grid_search.best_params_
# hgbcbest = HistGradientBoostingClassifier(**best_params)

# hgbcbest.fit(x_train, y_train)

# accuracy = hgbcbest.score(x_test, y_test)

# y_pred = hgbcbest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision= ' , precision_score(y_test,y_pred,average='macro'))

# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=hgbcbest.classes_)
# disp.plot() 
# title = 'Histogram-Based Gradient Boosting'
# plt.show()

# plot_learning_curve(hgbcbest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina Histogram-Based Gradient Boosting
################################################################################################################################
# AdaBoost

# param_grid = {
#     'n_estimators': [20, 50, 100], 
#     'learning_rate': [0.01, 0.1, 0.001],
#     'random_state' : [20, 70, 90]
# }

# adab = AdaBoostClassifier()

# grid_search = GridSearchCV(adab, param_grid=param_grid, cv=5, n_jobs=-1)

# grid_search.fit(x_train, y_train)

# best_params = grid_search.best_params_

# print(best_params)

# adabest = AdaBoostClassifier(**best_params)

# adabest.fit(x_train, y_train)

# scores = cross_val_score(adabest, x_train, y_train, cv=5) 
# print("Puntuaciones de validación cruzada:", scores)
# print("Puntuación media:", scores.mean())

# accuracy = adabest.score(x_test, y_test)
# y_pred = adabest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision= ' , precision_score(y_test,y_pred,average='macro'))

# print()

# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=adabest.classes_)
# disp.plot() 
# title = 'AdaBoost'
# plt.show()

# plot_learning_curve(adabest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina AdaBoost
################################################################################################################################
# Multi-layer Perceptron 

# param_grid = {
#     'hidden_layer_sizes': [(32,), (64,), (128,), (256,)],  
#     'activation': ['relu', 'tanh'],  
#     'solver': ['adam', 'sgd'],  
#     'alpha': [0.0001, 0.00001],  
#     'max_iter': [3000, 4000, 5000],  
# }

# mlp = MLPClassifier()

# grid_search = GridSearchCV(mlp, param_grid=param_grid, cv=5, n_jobs=-1)

# grid_search.fit(x_train, y_train)

# best_params = grid_search.best_params_

# print(best_params)

# mlpbest = MLPClassifier(**best_params)

# mlpbest.fit(x_train, y_train)

# scores = cross_val_score(mlpbest, x_train, y_train, cv=5) 
# print("Puntuaciones de validación cruzada:", scores)
# print("Puntuación media:", scores.mean())

# accuracy = mlpbest.score(x_test, y_test)
# y_pred = mlpbest.predict(x_test)

# print('accuracy =' , accuracy_score(y_test,y_pred))
# print('precision= ' , precision_score(y_test,y_pred,average='macro'))

# print()

# matriz=confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=mlpbest.classes_)
# disp.plot() 
# title = 'Multi-layer Perceptron'
# plt.show()

# plot_learning_curve(mlpbest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()