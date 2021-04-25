import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json



app = Flask(__name__)


model = pickle.load(open('RF.pkl', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('index.html', results={})

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    Predictors= [x for x in request.form.values()]
    TargetVariable='Target'
    f = set([ 'X' + str(i) for i in range(1,20)])
    if ( not f):
        
        return render_template('index.html', prediction_text='error in choice: {}') # rendering the predicted result

    X=DataForML[Predictors].values
    y=DataForML[TargetVariable].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)
    #prediction = model.predict(final_features) # making prediction

# Adaboost 

    # Choosing Decision Tree with 1 level as the weak learner
    # Choose different values of max_depth, n_estimators and learning_rate to tune the model
    DTC=DecisionTreeClassifier(max_depth=4)
    clf = AdaBoostClassifier(n_estimators=200, base_estimator=DTC ,learning_rate=0.01)

    # Printing all the parameters of Adaboost

    # Creating the model on Training Data
    AB=clf.fit(X_train,y_train)
    prediction=AB.predict(X_test)

    # Measuring accuracy on Testing Data
    from sklearn import metrics
    tree_metrics = metrics.classification_report(y_test, prediction,output_dict=True)['1']
    tree_metrics.pop('support',None)
    tree_metrics['accuracy'] = metrics.accuracy_score(y_test, prediction)
    for item in tree_metrics:
        tree_metrics[item] = round(tree_metrics[item], 2)
    r = {}
    r['decision tree'] = tree_metrics


    from sklearn.linear_model import LogisticRegression
    # choose parameter Penalty='l1' or C=1
    # choose different values for solver 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    clf = LogisticRegression(C=1,penalty='l2', solver='newton-cg')

    # Printing all the parameters of logistic regression
    # print(clf)

    # Creating the model on Training Data
    LOG=clf.fit(X_train,y_train)
    prediction=LOG.predict(X_test)

    # Measuring accuracy on Testing Data

    logistic_metrics = metrics.classification_report(y_test, prediction,output_dict=True)['1']
    logistic_metrics.pop('support',None)
    logistic_metrics['accuracy'] = metrics.accuracy_score(y_test, prediction)
    for item in logistic_metrics:
        logistic_metrics[item] = round(logistic_metrics[item], 2)
    r['logistic regression'] = logistic_metrics


    from sklearn.ensemble import RandomForestClassifier
    # Choose various values of max_depth, n_estimators and criterion for tuning the model
    clf = RandomForestClassifier(max_depth=10, n_estimators=100,criterion='gini')


    forest_metrics = metrics.classification_report(y_test, prediction,output_dict=True)['1']
    forest_metrics.pop('support',None)
    forest_metrics['accuracy'] = metrics.accuracy_score(y_test, prediction)
    for item in forest_metrics:
        forest_metrics[item] = round(forest_metrics[item], 2)
    r['random forest'] = forest_metrics


# Printing all the parameters of Random Forest
    from sklearn import svm
    clf = svm.SVC(C=2, kernel='rbf', gamma=0.1)

    # Printing all the parameters of KNN
    print(clf)

    # Creating the model on Training Data
    SVM=clf.fit(X_train,y_train)
    prediction=SVM.predict(X_test)

    svm_metrics = metrics.classification_report(y_test, prediction,output_dict=True)['1']
    svm_metrics.pop('support',None)
    svm_metrics['accuracy'] = metrics.accuracy_score(y_test, prediction)
    for item in svm_metrics:
        
        svm_metrics[item] = round(svm_metrics[item], 2)
    r['SVM'] = svm_metrics

    # Plotting the feature importance for Top 10 most important columns
    print('r',r)
    r = json.dumps(r)
    r = json.loads(r)






    return render_template('index.html', results=r) # rendering the predicted result

if __name__ == "__main__":
    app.run()
