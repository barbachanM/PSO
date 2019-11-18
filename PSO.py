


from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import inspect

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=LineSearchWarning)
seed = 1075
np.random.seed(seed)


expData = pd.read_csv("../expData_PSO.csv", index_col = 0)

y = expData['Group'].values

yDF = expData['Group']

X = expData.drop('Group', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=seed, stratify=y)




iter_max = 25
pop_size = 15
c1 = 1.5
c2 = 1.5
w = 1.2

## original
# w = 0.729
# c1 = c2 = 1.494

def errorFunction(classifier):
    classifierFit = classifier.fit(X_train,y_train)
    predictions = classifierFit.predict_proba(X_test)
    score = log_loss(y_test, predictions)
    return score
 
#initialize the particles

hyperPar = []
for i in range(pop_size):
    particle = {'RF':np.random.randint(10,150), 'SVM':np.random.uniform(1.0,7.0), 'LogReg':[np.random.uniform(1.0,7.0),np.random.randint(10,100)], 'KNN':np.random.randint(5,50)} #hyperparam
    hyperPar.append(particle)
swarm = []
for i in range(pop_size):
    p = {}
    p[0] = hyperPar[i] #hyperparameters
    p[1] = 99999999  #fitness
    p[2] = 0.0 #velocity
    p[3] = p[0] #best
    swarm.append(p)

out = open("PSO_logloss.csv", "w")
print(swarm[0])
j = 0
# let the first particle be the global best
gbest = swarm[0]
while j < iter_max :

    print("-----> "+str(j))
    for p in swarm:
        rf = RandomForestClassifier(n_estimators=p[0]['RF'],random_state=seed, bootstrap = True)
        svm = SVC(kernel='poly', gamma = 'auto',C =p[0]['SVM'] ,probability=True, random_state=seed)
        lg = LogisticRegression(solver = 'newton-cg',max_iter = p[0]['LogReg'][1],C =p[0]['LogReg'][0], random_state=seed)
        knn = KNeighborsClassifier(n_neighbors = p[0]['KNN'])
        lda = LinearDiscriminantAnalysis(solver = 'svd')
        vote = VotingClassifier(estimators=[('SVM', svm), ('Random Forests', rf), ('LogReg', lg), ('KNN', knn), ('LDA',lda)], voting='soft')
        fitness = errorFunction(vote)
        print(fitness)
        if fitness < gbest[1]:

            print('\n*** Global Best! '+str(fitness)+"\n")
            out.write(str(j)+","+str(fitness)+"\n")
            gbest = p
        if fitness < p[1]:
            print("--- Local Best! "+str(fitness))
            p[1] = fitness
            p[3] = p[0] 
        else:
            p[1] = fitness
        for clf in p[0].keys():
            if clf == 'LogReg':
                ## max iter
                v = w*p[2] + c1 * np.random.uniform(0,1) * (p[3][clf][0] - p[0][clf][0]) + c2 * np.random.uniform(0,1) * (gbest[3][clf][0] - p[3][clf][0])  
                p[0][clf][0] = abs(p[0][clf][0] + round(v))
                ## C
                v = round(w*p[2] + c1 * np.random.uniform(0,1) * (p[3][clf][1] - p[0][clf][1]) + c2 * np.random.uniform(0,1) * (gbest[3][clf][1] - p[3][clf][1]))
                p[0][clf][1] = abs(p[0][clf][1] + v) 
            elif clf == 'RF' or clf == 'KNN' :
                ## n_estimarors (RF) n_neighbors (KNN)
                v = w*p[2] + c1 * np.random.uniform(0,1) * (p[3][clf] - p[0][clf]) + c2 * np.random.uniform(0,1) * (gbest[3][clf] - p[3][clf])
                p[0][clf] = abs(p[0][clf] + round(v))

            else:#C (SVM)
                v = w*p[2] + c1 * np.random.uniform(0,1) * (p[3][clf] - p[0][clf]) + c2 * np.random.uniform(0,1) * (gbest[3][clf] - p[3][clf])
                p[0][clf] = abs(p[0][clf] + v)


          
    j  += 1



##0.31850293439489524 iteration 1 w = 1.2, c1=2, c2 = 2.5 np = 100 niter = 10
##0.31850293439489524
##0.3114929365890441 
##0.3099700599879927
##0.3066713979393286/0.3218361738219337
##0.3066467105635484

