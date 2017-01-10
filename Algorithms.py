# ignoring gender confidence
# give reasons through code

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import scipy as sp
from sklearn.metrics import roc_curve, auc
import Utilities
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_predict
from sklearn import cross_validation
import random
from sklearn.metrics import accuracy_score

#  testing Naive Bayes
def testing_NaiveBayes(data_frame):
    # 2: Male, 1: female, 0: brand
    print("Algorithm chosen: Naive Bayes")
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    x = vectorizer.fit_transform(data_frame['valid_words'],data_frame['sidebar_color'])
    x1 =  vectorizer.fit_transform(data_frame['sidebar_color'], data_frame['link_color'])        # creating feature vector for side bar color
    X = sp.sparse.hstack((x, x1), format='csr')
    #print(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(data_frame['gender'])
    #print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    print("Accuracy without k-cross validation", nb.score(x_test, y_test) * 100)
    multinomial_nb_with_cv(x_train, y_train)

    algorithm_evaluation(nb, "Naive Bayes", x_test, y_test)

# testing Naive Bayes with cv
def multinomial_nb_with_cv(x_train, y_train):
    #random.shuffle(X)
    kf = cross_validation.KFold(x_train.shape[0], n_folds=10)
    acc = []
    for train_index, test_index in kf:
        y_true = y_train[test_index]
        clf = MultinomialNB().fit(x_train[train_index],
                                  y_train[train_index])
        y_pred = clf.predict(x_train[test_index])
        acc.append(accuracy_score(y_true, y_pred))
    #print(acc)
    print("Accuracy with cross validation", reduce(lambda x, y: x + y, acc) / len(acc) * 100)

# testing Random forest
def testing_randomForests(data_frame):
    print("Algorithm chosen: Random Forest")
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    x = vectorizer.fit_transform(data_frame['valid_words'])
    x1 = vectorizer.fit_transform(data_frame['sidebar_color'], data_frame['link_color'])  # creating feature vector for side bar color
    X = sp.sparse.hstack((x, x1), format='csr')
    X = X.toarray()

    # x1 =  vectorizer_color.transform(test_dataFrame['sidebar_color'], test_dataFrame['link_color'])        # creating feature vector for side bar color
    encoder = LabelEncoder()
    y = encoder.fit_transform(data_frame['gender'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(x_train, y_train)
    print("Accuracy without k-cross validation", forest.score(x_test, y_test) * 100)
    random_forest_with_cv(x_train, y_train)
    algorithm_evaluation(forest, "Random Forest", x_test, y_test)

# testing random forest with cv
def random_forest_with_cv(x_train, y_train):
    #random.shuffle(X)
    kf = cross_validation.KFold(x_train.shape[0], n_folds=10)
    acc = []
    for train_index, test_index in kf:
        y_true = y_train[test_index]
        clf = RandomForestClassifier(n_estimators=100).fit(x_train[train_index],
                                  y_train[train_index])
        y_pred = clf.predict(x_train[test_index])
        acc.append(accuracy_score(y_true, y_pred))
    #print(acc)
    print("Accuracy with cross validation", reduce(lambda x, y: x + y, acc) / len(acc) * 100)


# testing Logistic Regression
def testing_logsiticRegression(data_frame):
    print("Algorithm chosen: Logistic Regression")
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    x = vectorizer.fit_transform(data_frame['valid_words'], data_frame['sidebar_color'])
    x1 = vectorizer.fit_transform(data_frame['sidebar_color'],
                                  data_frame['link_color'])  # creating feature vector for side bar color
    X = sp.sparse.hstack((x, x1), format='csr')
    # print(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(data_frame['gender'])
    # print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    print("Accuracy without k-cross validation", model.score(x_test, y_test) * 100)
    logisticRegression_with_cv(x_train, y_train)
    algorithm_evaluation(model, "Logistic Regression", x_test, y_test)

# testing Logistic Regression with cv
def logisticRegression_with_cv(x_train, y_train):
    #random.shuffle(X)
    kf = cross_validation.KFold(x_train.shape[0], n_folds=10)
    acc = []
    for train_index, test_index in kf:
        y_true = y_train[test_index]
        clf = LogisticRegression().fit(x_train[train_index],
                                  y_train[train_index])
        y_pred = clf.predict(x_train[test_index])
        acc.append(accuracy_score(y_true, y_pred))
    #print(acc)
    print("Accuracy with cross validation", reduce(lambda x, y: x + y, acc) / len(acc) * 100)

def testing_supportVectorMachines(data_frame):
    print("Algorithm chosen: Linear SVM")
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    x = vectorizer.fit_transform(data_frame['valid_words'], data_frame['sidebar_color'])
    x1 = vectorizer.fit_transform(data_frame['sidebar_color'],
                                  data_frame['link_color'])  # creating feature vector for side bar color
    X = sp.sparse.hstack((x, x1), format='csr')
    # print(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(data_frame['gender'])
    # print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_svc = SVC(kernel='linear', gamma=1, C=1)
    model_svc.fit(x_train, y_train)
    print(model_svc.score(x_test, y_test) * 100)
    predictions = model_svc.predict(x_test)
    print '\nConfussion matrix:\n', confusion_matrix(y_test, predictions)
    algorithm_evaluation(model_svc, "Linear SVM", x_test, y_test)



def algorithm_evaluation(model, algorithm_label, x_test, y_test):
    predictions = model.predict(x_test)
    if(algorithm_label == 'Linear SVM'):
        scores = model.decision_function(x_test)
        #print(scores)
    else:
        scores = model.predict_proba(x_test)
    scores_0 = [item[0] for item in scores]
    scores_1 = [item[1] for item in scores]
    scores_2 = [item[2] for item in scores]

    #print(scores_0)
    # analysing Naive Bayes
    false_positive_rate_score0, true_positive_rate_score0, thresholds_score0 = roc_curve(y_test, scores_0, pos_label=0)
    Utilities.plot_ROC_curve(false_positive_rate_score0, true_positive_rate_score0, "Gender = brands", algorithm_label)
    roc_auc_score0 = auc(false_positive_rate_score0, true_positive_rate_score0)
    print("Area under the curve for brands ", roc_auc_score0)

    false_positive_rate_score1, true_positive_rate_score1, thresholds_score1 = roc_curve(y_test, scores_1, pos_label=1)
    Utilities.plot_ROC_curve(false_positive_rate_score1, true_positive_rate_score1, "Gender = female", algorithm_label)
    roc_auc_score1 = auc(false_positive_rate_score1, true_positive_rate_score1)
    print("Area under the curve for female ", roc_auc_score1)

    false_positive_rate_score2, true_positive_rate_score2, thresholds_score2 = roc_curve(y_test, scores_2, pos_label=2)
    Utilities.plot_ROC_curve(false_positive_rate_score2, true_positive_rate_score2, "Gender = male", algorithm_label)
    roc_auc_score2 = auc(false_positive_rate_score2, true_positive_rate_score2)
    print("Area under the curve for male ", roc_auc_score2)

    # Confusion matrix
    print '\nClassification report:\n', classification_report(y_test, predictions)
    print '\nConfusion matrix:\n', confusion_matrix(y_test, predictions)


