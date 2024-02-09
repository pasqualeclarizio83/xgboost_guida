# Reti Neurali e XgBoost

Breve guida sulle Reti Neurali

## Descrizione

XGBoost è una libreria open-source molto popolare per l'apprendimento automatico, specialmente per problemi di regressione e classificazione. Si basa sulla tecnica di boosting, che è un metodo di ensemble learning in cui vengono combinati diversi modelli più deboli per creare un modello più forte.

Ecco una breve panoramica di ciò che c'è da sapere su XGBoost:

Boosting: XGBoost è basato sul concetto di boosting, dove vengono addestrati numerosi modelli di base, chiamati "weak learners" o "base learners", in sequenza. Ad ogni passaggio, il modello si concentra sugli esempi che sono stati classificati in modo errato o che sono stati difficili da classificare, fornendo loro un peso maggiore nell'addestramento del modello successivo.

Alberi Decisionali: XGBoost utilizza alberi decisionali come base learners. Gli alberi decisionali sono modelli di apprendimento automatico che prendono decisioni basate su una serie di domande sui dati di input.

Regolarizzazione: XGBoost utilizza una varietà di tecniche di regolarizzazione per prevenire l'overfitting, come la regolarizzazione L1 e L2 e il pruning degli alberi.

Funzione di Perdita Personalizzabile: XGBoost consente di specificare una vasta gamma di funzioni di perdita (loss function) in base al tipo di problema che si sta affrontando, come la regressione lineare, la regressione logistica, la classificazione multiclasse, ecc.

Parallelizzazione e Ottimizzazione: XGBoost è ottimizzato per massimizzare le prestazioni e può sfruttare al massimo le risorse hardware disponibili, come la parallelizzazione su più core della CPU e l'utilizzo della GPU per l'addestramento.


```bash

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Caricamento del dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Creazione del modello XGBoost
model = xgb.XGBClassifier()

# Addestramento del modello
model.fit(X_train, y_train)

# Predizione
y_pred = model.predict(X_test)

# Valutazione
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


```


## Un esempio di Albero decisionale con xgBoost

```bash

import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Creazione di un dataset di esempio
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definizione dei parametri del modello
params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}

# Addestramento del modello XGBoost
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Visualizzazione dell'albero decisionale
plt.figure(figsize=(20, 10))
plot_tree(model, num_trees=0, rankdir='LR')
plt.show()

```




## Dati in input le caratteristiche di un animale
## in output

```bash

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Definizione del dataset di esempio
# Supponiamo che X sia una matrice che contiene le caratteristiche degli animali e y sia un array che contiene le etichette (0 per cane, 1 per gatto)

# Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definizione dei parametri del modello XGBoost
params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}

# Addestramento del modello XGBoost
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Predizione
y_pred = model.predict(X_test)

# Valutazione
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


```




## Dati in input le caratteristiche di un animale con degli esempi
## in output e otteniamo in output se 
## l'animale è un cane o gatto

```bash

import numpy as np
import xgboost as xgb

# Caratteristiche di un nuovo animale (esempio)
# Supponiamo che le caratteristiche siano: altezza, peso, lunghezza del pelo, ecc.
new_animal_features = np.array([[50, 10, 2], [30, 5, 1]])  # Esempio di due nuovi animali

# Creazione del modello XGBoost
params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = xgb.XGBClassifier(**params)

# Addestramento del modello con dati di esempio
X_train = np.array([[45, 12, 3], [35, 8, 2], [40, 10, 2], [25, 5, 1], [30, 7, 2], [20, 4, 1]])  # Esempio di caratteristiche di training
y_train = np.array([0, 0, 0, 1, 1, 1])  # Esempio di etichette (0 per cane, 1 per gatto)
model.fit(X_train, y_train)

# Predizione sulle caratteristiche del nuovo animale
predictions = model.predict(new_animal_features)

# Stampa delle predizioni
for i, pred in enumerate(predictions):
    if pred == 0:
        print(f"L'animale {i+1} è previsto essere un cane.")
    else:
        print(f"L'animale {i+1} è previsto essere un gatto.")


```




## Dati in input le caratteristiche di un animale con degli esempi
## in output e otteniamo in output se 
## l'animale è un cane o gatto o coniglio


```bash

Preparazione dei dati: Assicurati di avere un set di dati che includa le caratteristiche degli animali e le relative etichette (per esempio, 0 per cane, 1 per gatto, 2 per coniglio). Le caratteristiche degli animali devono essere rappresentate in forma numerica.

Creazione del modello: Utilizziamo XGBoost per creare un modello di classificazione. Definiamo i parametri del modello e addestriamolo utilizzando i dati preparati.

Predizione: Utilizziamo il modello addestrato per fare predizioni sulle caratteristiche di un nuovo animale.

```

```bash

import numpy as np
import xgboost as xgb

# Caratteristiche di un nuovo animale (esempio)
# Supponiamo che le caratteristiche siano: altezza, peso, lunghezza del pelo, ecc.
new_animal_features = np.array([[50, 10, 2], [30, 5, 1]])  # Esempio di due nuovi animali

# Creazione del modello XGBoost
params = {'objective': 'multi:softmax', 'num_class': 3, 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = xgb.XGBClassifier(**params)

# Addestramento del modello con dati di esempio
X_train = np.array([[45, 12, 3], [35, 8, 2], [40, 10, 2], [25, 5, 1], [30, 7, 2], [20, 4, 1]])  # Esempio di caratteristiche di training
y_train = np.array([0, 0, 0, 1, 1, 2])  # Esempio di etichette (0 per cane, 1 per gatto, 2 per coniglio)
model.fit(X_train, y_train)

# Predizione sulle caratteristiche del nuovo animale
predictions = model.predict(new_animal_features)

# Stampa delle predizioni
for i, pred in enumerate(predictions):
    if pred == 0:
        print(f"L'animale {i+1} è previsto essere un cane.")
    elif pred == 1:
        print(f"L'animale {i+1} è previsto essere un gatto.")
    else:
        print(f"L'animale {i+1} è previsto essere un coniglio.")


```






## Supponiamo di avere un set di dati che contiene le 
## caratteristiche degli animali e le relative etichette 
## di classe (0 per cane, 1 per gatto, 2 per coniglio).


```bash

import numpy as np
import xgboost as xgb

# Caratteristiche degli animali (altezza, peso, lunghezza del pelo)
# Supponiamo che ognuna delle tre caratteristiche sia rappresentata da un valore numerico
animal_features = np.array([
    [45, 12, 3],  # Cane
    [35, 8, 2],   # Cane
    [40, 10, 2],  # Cane
    [25, 5, 1],   # Gatto
    [30, 7, 2],   # Gatto
    [20, 4, 1]    # Coniglio
])

# Etichette di classe (0 per cane, 1 per gatto, 2 per coniglio)
labels = np.array([0, 0, 0, 1, 1, 2])

# Creazione del modello XGBoost
params = {'objective': 'multi:softmax', 'num_class': 3, 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = xgb.XGBClassifier(**params)

# Addestramento del modello
model.fit(animal_features, labels)

# Predizione per nuovi animali
new_animal_features = np.array([
    [50, 10, 2],  # Nuovo animale
    [30, 5, 1]    # Nuovo animale
])

predictions = model.predict(new_animal_features)

# Stampa delle predizioni
for i, pred in enumerate(predictions):
    if pred == 0:
        print(f"L'animale {i+1} è previsto essere un cane.")
    elif pred == 1:
        print(f"L'animale {i+1} è previsto essere un gatto.")
    else:
        print(f"L'animale {i+1} è previsto essere un coniglio.")

```




## XgBoost - Classificazione
## un esempio simile al precedente


```bash

import numpy as np
import xgboost as xgb

# Definizione delle caratteristiche degli animali (altezza, peso, lunghezza del pelo)
animal_features = np.array([
    [45, 12, 3],  # Cane
    [35, 8, 2],   # Cane
    [40, 10, 2],  # Cane
    [25, 5, 1],   # Gatto
    [30, 7, 2],   # Gatto
    [20, 4, 1]    # Coniglio
])

# Etichette di classe (0 per cane, 1 per gatto, 2 per coniglio)
labels = np.array([0, 0, 0, 1, 1, 2])

# Creazione e addestramento del modello XGBoost
params = {'objective': 'multi:softmax', 'num_class': 3, 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = xgb.XGBClassifier(**params)
model.fit(animal_features, labels)

# Definizione delle caratteristiche di nuovi animali
new_animals = np.array([
    [50, 10, 2],  # Nuovo animale
    [30, 5, 1]    # Nuovo animale
])

# Predizione delle classi dei nuovi animali
predictions = model.predict(new_animals)

# Mappatura delle predizioni alle etichette di classe
class_mapping = {0: 'Cane', 1: 'Gatto', 2: 'Coniglio'}
predicted_classes = [class_mapping[pred] for pred in predictions]

# Stampa delle predizioni
for i, animal in enumerate(predicted_classes):
    print(f"Animale {i+1}: {animal}")


```




## XgBoost - Classificazione
## dato in ingresso le caratteristiche di una Persona
## determina in output se è Uomo o Donna


```bash

import numpy as np
import xgboost as xgb

# Caratteristiche delle persone (esempio: altezza in cm, peso in kg, età in anni)
people_features = np.array([
    [180, 80, 30],  # Uomo
    [165, 60, 25],  # Donna
    [175, 70, 40],  # Uomo
    [160, 55, 35],  # Donna
    [185, 90, 45]   # Uomo
])

# Etichette di classe (0 per uomo, 1 per donna)
labels = np.array([0, 1, 0, 1, 0])

# Creazione e addestramento del modello XGBoost
params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = xgb.XGBClassifier(**params)
model.fit(people_features, labels)

# Definizione delle caratteristiche di una nuova persona
new_person = np.array([
    [170, 65, 28]   # Nuova persona
])

# Predizione della classe della nuova persona
prediction = model.predict(new_person)

# Mappatura della predizione alla classe
class_mapping = {0: 'Uomo', 1: 'Donna'}
predicted_class = class_mapping[prediction[0]]

# Stampa della predizione
print(f"La persona è prevista essere: {predicted_class}")


```




## XgBoost - Classificazione
## dato in ingresso le caratteristiche di uno sviluppatore
## determina in output se più portato per l'analisi o lo sviluppo


```bash

import numpy as np
import xgboost as xgb

# Caratteristiche degli sviluppatori (esempio: esperienza in anni, competenze tecniche, preferenze di lavoro)
developers_features = np.array([
    [5, 8, 3],   # Più portato all'analisi
    [3, 7, 2],   # Più portato allo sviluppo
    [8, 9, 4],   # Più portato all'analisi
    [4, 6, 2],   # Più portato allo sviluppo
    [6, 8, 3]    # Più portato all'analisi
])

# Etichette di classe (0 per più portato all'analisi, 1 per più portato allo sviluppo)
labels = np.array([0, 1, 0, 1, 0])

# Creazione e addestramento del modello XGBoost
params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = xgb.XGBClassifier(**params)
model.fit(developers_features, labels)

# Definizione delle caratteristiche di uno sviluppatore
new_developer = np.array([
    [4, 7, 3]   # Nuovo sviluppatore
])

# Predizione della classe dello sviluppatore
prediction = model.predict(new_developer)

# Mappatura della predizione alla classe
class_mapping = {0: 'Più portato all\'analisi', 1: 'Più portato allo sviluppo'}
predicted_class = class_mapping[prediction[0]]

# Stampa della predizione
print(f"Lo sviluppatore è previsto essere: {predicted_class}")


```





## XgBoost - Regressione
## dato in input le dimensioni di un immobile
## e otteniamo in output l'andamento del prezzo

```bash

Preparazione dei dati: Raccolta di dati che includano le dimensioni di vari immobili e i loro prezzi di vendita. Le dimensioni dell'immobile devono essere rappresentate in forma numerica.

Creazione del modello: Utilizzo di XGBoost per creare un modello di regressione in modo da predire il prezzo dell'immobile basandosi sulle sue dimensioni.

Addestramento del modello: Addestramento del modello utilizzando i dati di allenamento per apprendere la relazione tra le dimensioni dell'immobile e il prezzo di vendita.

Predizione: Utilizzo del modello addestrato per fare predizioni sul prezzo di vendita di un nuovo immobile basandosi sulle sue dimensioni.


```


```bash

import numpy as np
import xgboost as xgb

# Caratteristiche degli immobili (esempio: dimensioni in metri quadrati)
house_sizes = np.array([
    [100],   # Prima casa
    [150],   # Seconda casa
    [120],   # Terza casa
    [200],   # Quarta casa
    [80]     # Quinta casa
])

# Prezzi di vendita degli immobili corrispondenti
prices = np.array([300000, 400000, 350000, 500000, 250000])

# Creazione e addestramento del modello XGBoost
params = {'objective': 'reg:squarederror', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = xgb.XGBRegressor(**params)
model.fit(house_sizes, prices)

# Definizione delle dimensioni di un nuovo immobile
new_house_size = np.array([
    [180]   # Nuova casa
])

# Predizione del prezzo di vendita del nuovo immobile
predicted_price = model.predict(new_house_size)

# Stampa della predizione
print(f"Prezzo previsto per il nuovo immobile: {predicted_price[0]}")



```





## XgBoost - Regressione
## dato in input le dimensioni di un immobile
## e otteniamo in output l'andamento del prezzo


```bash

import numpy as np
import xgboost as xgb

# Caratteristiche delle case e paesi di provenienza (esempio: dimensioni, numero di camere, paese)
# Supponiamo che il paese sia rappresentato tramite codifica one-hot (es. [1, 0] per Italia, [0, 1] per Francia)
house_features = np.array([
    [100, 3, 1, 0],   # Italia
    [120, 4, 0, 1],   # Francia
    [150, 5, 1, 0],   # Italia
    [80, 2, 0, 1],    # Francia
    [200, 6, 1, 0]    # Italia
])

# Prezzi di vendita delle case corrispondenti
prices = np.array([300000, 400000, 350000, 250000, 500000])

# Creazione e addestramento del modello XGBoost
params = {'objective': 'reg:squarederror', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = xgb.XGBRegressor(**params)
model.fit(house_features, prices)

# Definizione delle caratteristiche di una nuova casa (esempio: dimensioni, numero di camere, paese)
new_house_features = np.array([
    [180, 4, 1, 0]    # Italia
])

# Predizione del prezzo di vendita della nuova casa
predicted_price = model.predict(new_house_features)

# Stampa della predizione
print(f"Prezzo previsto per la nuova casa: {predicted_price[0]}")

```