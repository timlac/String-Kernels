#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from preprocessing import data, get_all_classes, process_directory
from postprocessing import evaluate, evaulate_multiple_runs, print_results


def train_target(train_classes, filter_classes=[]):
    """make classes into vector for training"""

    if filter_classes:
        class_set = set(filter_classes)
    else:
        class_set = get_all_classes()

    mlb = MultiLabelBinarizer(classes=list(class_set))
    y_train = mlb.fit_transform(train_classes)

    return y_train, mlb


def train(train_text_list,
          train_classes_list,
          n_features=3000,
          filter_classes=[]):

    vectorizer = TfidfVectorizer(
        max_features=n_features,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True)

    X_train = vectorizer.fit_transform(train_text_list)
    y_train, mlb = train_target(train_classes_list, filter_classes)

    classifier = OneVsRestClassifier(SVC(kernel='linear'))
    classifier.fit(X_train, y_train)
    return classifier, vectorizer, mlb


def test(test_text_list, test_classes_list, classifier, vectorizer,
         multilabelbinarizer):

    X_test = vectorizer.transform(test_text_list)
    y_test = multilabelbinarizer.transform(test_classes_list)
    y_pred = classifier.predict(X_test)
    return y_test, y_pred


def main(n_runs, n_train_samples, n_test_samples, n_features, filter_classes):

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    train_ids, test_ids, _, texts, classes = process_directory()

    for run in range(n_runs):

        train_texts, train_classes, test_texts, test_classes = data(
            train_ids, test_ids, texts, classes, n_train_samples,
            n_test_samples, filter_classes)

        classifier, vectorizer, mlb = train(train_texts, train_classes,
                                            n_features, filter_classes)

        y_test, y_pred = test(test_texts, test_classes, classifier, vectorizer,
                              mlb)

        p, r, f1, s = evaluate(y_test, y_pred, mlb)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        supports.append(s)

    return evaulate_multiple_runs(n_runs, filter_classes, precisions, recalls,
                                  f1_scores, supports)


if __name__ == '__main__':
    r, h = main(2, 380, 90, 1000, ['earn', 'acq', 'crude', 'corn'])
    print(print_results(r, h))
