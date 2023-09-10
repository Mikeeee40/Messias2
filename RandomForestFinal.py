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

##ENCODING
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

df_majority = Messias[Messias['victoria'] == 1]
df_minority = Messias[Messias['victoria'] == 0]

n_samples_to_keep = 90  # Número deseado de instancias con 'target' igual a 0
df_majority_subsampled = resample(df_majority, replace=False, n_samples=n_samples_to_keep, random_state=42)

Messias = pd.concat([df_majority_subsampled, df_minority])

print(Messias['victoria'].value_counts())
      
################################################################################################################################
# Termina limpieza de datos y empieza modelos
################################################################################################################################
# Empieza la separación en Train y Test y se realiza un Random Search

x = Messias.drop('victoria', axis = 1)
y = Messias['victoria']

x_train, x_test ,y_train, y_test = train_test_split( x , y , random_state=10)

################################################################################################################################
# Termina la separación en Train y Test
################################################################################################################################

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
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

def mean_squared_log_error(y_true, y_pred):
    y_true = np.log1p(y_true)
    y_pred = np.log1p(y_pred)
    return np.mean((y_true - y_pred) ** 2)
msle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)


################################################################################################################################
# Termina función para plotear línea de Aprendizaje 
################################################################################################################################
# Random Forest con Random Search

param_dist = {
    'n_estimators': [90, 95, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [48, 49, 50, 51, 52, 53, 54, None],
    'min_samples_split': [5, 6, 7, 8, 9],
    'min_samples_leaf': [3, 4, 5, 6],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight' : ['balanced']
}

rfc = RandomForestClassifier()

random_search = RandomizedSearchCV(rfc, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)

random_search.fit(x_train, y_train)

best_params = random_search.best_params_
print(best_params)

rfcbest = RandomForestClassifier(**best_params)

rfcbest.fit(x_train, y_train)

param_range = np.arange(1, 11)

train_scores, test_scores = validation_curve(RandomForestClassifier(**best_params), x_train, y_train, 
                                             param_name="max_depth", param_range=param_range, cv=5, scoring="accuracy")

scores = cross_val_score(rfcbest, x_train, y_train, cv=5, scoring=msle_scorer) 
#print("Puntuaciones de validación cruzada:", scores)
#print("Puntuación media:", scores.mean())
print("Mean Squared Logarithmic Error:", -np.mean(scores))

accuracy = rfcbest.score(x_test, y_test)
y_pred = rfcbest.predict(x_test)

print('accuracy =' , accuracy_score(y_test,y_pred))
print('precision= ' , precision_score(y_test,y_pred,average='macro'))

matriz=confusion_matrix(y_test,y_pred)
plt.rcParams["figure.dpi"] = 200
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=rfcbest.classes_)
disp.plot()

title = "Random Forest"

plt.show()

plot_learning_curve(rfcbest, title, x_train, y_train, cv=4, n_jobs=-1)
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

plt.title("Validation Curve for Decision Tree")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.xticks(param_range)
plt.legend(loc="lower right")
plt.grid()
plt.show()

################################################################################################################################
# Termina Random Forest con Random Search
################################################################################################################################
# Random Forest con Grid Search

# param_grid = {
#     'n_estimators': [90, 95, 100],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [48, 49, 50, 51],
#     'min_samples_split': [5, 6, 7, 8],
#     'min_samples_leaf': [3, 4, 5, 6],
#     'max_features': ['auto', 'sqrt', 'log2', None],
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

# matriz = confusion_matrix(y_test,y_pred)
# plt.rcParams["figure.dpi"] = 200
# disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=rfcbest.classes_)
# disp.plot() 
# title = 'Random Forest'
# plt.show()

# plot_learning_curve(rfcbest, title, x_train, y_train, cv=4, n_jobs=-1)
# plt.show()

################################################################################################################################
# Termina Random Forest con Grid Search
################################################################################################################################