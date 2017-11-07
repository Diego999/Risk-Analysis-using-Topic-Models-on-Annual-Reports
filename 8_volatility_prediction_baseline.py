import os
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn import linear_model
from scipy.stats import randint as sp_randint
from scipy.stats import expon
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
import random
config = __import__('0_config')
visualize = __import__('4e_visualize')

TRAINING_SPLIT = 0.8
VALID_SPLIT = 0.2


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion(Y_testing, Y_hat, classes):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_testing, Y_hat)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Class {}'.format(k) for k in range(len(classes))], title=key + ' Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Class {}'.format(k) for k in range(len(classes))], normalize=True, title=key + ' Confusion matrix, with normalization')
    plt.draw()


def print_and_get_accuracy(Y_testing, Y_hat):
    res = 100 * accuracy_score(Y_testing, Y_hat)
    print('Accuracy : {}'.format('{:.2f}'.format(res)))
    return res


def print_and_get_precision_recall_fscore_support(Y_testing, Y_hat):
    precision, recall, fscore, support = score(Y_testing, Y_hat)
    print('Precision: {}'.format(['{:.2f}'.format(100 * x) for x in precision]))
    print('Recall   : {}'.format(['{:.2f}'.format(100 * x) for x in recall]))
    print('Fscore   : {}'.format(['{:.2f}'.format(100 * x) for x in fscore]))
    print('Support  : {}'.format(['{:.2f}'.format(100 * x) for x in support]))
    return precision, recall, fscore, support


def print_and_get_classification_report(Y_testing, Y_hat):
    res = classification_report(Y_testing, Y_hat, target_names=['Class {}'.format(k) for k in range(len(classes))], digits=4)
    print(res)
    return res

# NE pas repousser les cuticules. Seulement après la douche avec un ongle humide
# Toujours mettre des gants en faisant la vaisselle !
#90-120k cheveux. On perd de 10 à 60 par jour. Pousse 1cm par mois
def plot_roc(Y_testing, Y_hat, classes, key):
    Y_testing_onehot = label_binarize(Y_testing, classes)
    Y_hat_onehot = label_binarize(Y_hat, classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(Y_testing_onehot[:, i], Y_hat_onehot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_testing_onehot.ravel(), Y_hat_onehot.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(key + ' ROC')
    plt.legend(loc="lower right")
    plt.draw()


def plot_prec_rec_curve(Y_testing, Y_hat, classes, key):
    Y_testing_onehot = label_binarize(Y_testing, classes)
    Y_hat_onehot = label_binarize(Y_hat, classes)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(Y_testing_onehot[:, i], Y_hat_onehot[:, i])
        average_precision[i] = average_precision_score(Y_testing_onehot[:, i], Y_hat_onehot[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_testing_onehot.ravel(), Y_hat_onehot.ravel())
    average_precision["micro"] = average_precision_score(Y_testing_onehot, Y_hat_onehot, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    fig = plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(len(classes)), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    #fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(key + ' Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.draw()


def tune(clf, X, Y, param_dist, n_iter_search=3):
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=1, cv=3)
    random_search.fit(X, Y)
    report(random_search.cv_results_)
    return random_search.cv_results_


if __name__ == "__main__":
    sections_to_analyze = [config.DATA_1A_FOLDER]#, config.DATA_7A_FOLDER, config.DATA_7_FOLDER]

    for section in sections_to_analyze:
        for key in ['volatility', 'roe']:
            input_file = os.path.join(section[:section.rfind('/')], section[section.rfind('/')+1:] + config.SUFFIX_DF + '_{}.pkl'.format(key))
            data = pd.read_pickle(input_file)

            random.seed(config.SEED)

            data = [{'file':file, 'topics':[xx[1] for xx in x['topics']], 'label':x[key + '_label']} for file, x in data.iterrows()]
            classes = sorted(list({x['label'] for x in data}))
            mapping = {c: i for i, c in enumerate(classes)}
            for x in data:
                x['label'] = mapping[x['label']]

            random.shuffle(data)

            training_size = int(len(data)*TRAINING_SPLIT)
            validation_training_size = int(training_size*(1.0-VALID_SPLIT))
            training = data[:validation_training_size]
            validation = data[validation_training_size:training_size]
            testing = data[training_size:]

            X_training, Y_training = [x['topics'] for x in training], [x['label'] for x in training]
            X_valid, Y_valid = [x['topics'] for x in validation], [x['label'] for x in validation]
            X_testing, Y_testing = [x['topics'] for x in testing], [x['label'] for x in testing]

            #classifiers = [linear_model.LogisticRegression(solver='lbfgs'), RandomForestClassifier(n_estimators=20)]
            #param_dists = [{"C": expon(100),
            #              "multi_class": ["ovr", "multinomial"]},
            #               {"max_depth": [3, None],
            #              "max_features": sp_randint(1, 11),
            #              "min_samples_split": sp_randint(2, 11),
            #              "min_samples_leaf": sp_randint(1, 11),
            #              "bootstrap": [True, False],
            #              "criterion": ["gini", "entropy"]}]
            ## CV, no need of validation
            #for clf, param_dist in zip(classifiers, param_dists):
            #    res = tune(clf, X_training + X_valid, Y_training + Y_valid, param_dist)

            classifiers = [linear_model.LogisticRegression(solver='lbfgs'), RandomForestClassifier(n_estimators=20), SVC(kernel='linear'), SVC(kernel='rbf'), MLPClassifier(early_stopping=True)]
            for clf in classifiers:
                clf.fit(X_training+X_valid, Y_training+Y_valid)
                Y_hat = clf.predict(X_testing)
                print(key)
                print_and_get_accuracy(Y_testing, Y_hat)
                print_and_get_precision_recall_fscore_support(Y_testing, Y_hat)
                print_and_get_classification_report(Y_testing, Y_hat)
                #plot_confusion(Y_testing, Y_hat, classes)
                #plot_roc(Y_testing, Y_hat, classes, key)
                #plot_prec_rec_curve(Y_testing, Y_hat, classes, key)
                #plt.show()