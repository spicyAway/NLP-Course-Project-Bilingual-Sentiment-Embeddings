import sys
import os
import argparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from Utils.Datasets import *
from Utils.WordVecs import *
from Utils.utils import *

def scores(model, dataset):
    p = model.predict(dataset._Xtest)
    acc = accuracy_score(dataset._ytest, p)
    prec = per_class_prec(dataset._ytest, p).mean()
    rec = per_class_rec(dataset._ytest, p).mean()
    f1 = macro_f1(dataset._ytest, p)
    return acc, prec, rec, f1

def get_W(pdataset, src_vecs, trg_vecs):
    X, Y = [], []
    for i in pdataset._Xtrain:
        X.append(src_vecs[i])
    for i in pdataset._ytrain:
        Y.append(trg_vecs[i])

    X = np.array(X)
    Y = np.array(Y)
    u, s, vt = np.linalg.svd(np.dot(Y.T, X))
    W = np.dot(vt.T, u.T)
    return W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_vecs', default='embeddings/google.txt',
                        help=" source language vectors (default: GoogleNewsVecs )")
    parser.add_argument('-trg_vecs', default='embeddings/fr.txt',
                        help=" target language vectors - french")
    parser.add_argument('-trans', help='translation pairs',
                        default='lexicons/French/Combined.txt')
    parser.add_argument('-dataset', default='opener_sents', help="dataset to train and test on (default: opener)")
    parser.add_argument('-bi',
                        help='List of booleans. True is only binary, False is only 4 class. True False is both. (default: [True, False])',
                        default=[True, False], nargs='+', type=str2bool)
    args = parser.parse_args()

    # Loop over the three languages
    for lang in ['fr']:
        print('################ {0} ##############'.format(lang))

        # Import monolingual vectors
        print('importing word embeddings')
        src_vecs = WordVecs(args.src_vecs)
        src_vecs.mean_center()
        src_vecs.normalize()

        trg_vecs = WordVecs(args.trg_vecs)
        trg_vecs.mean_center()
        trg_vecs.normalize()

        # Setup projection dataset
        pdataset = ProjectionDataset(args.trans, src_vecs, trg_vecs)

        # learn the translation matrix W
        W = get_W(pdataset, src_vecs, trg_vecs)

        # project the source matrix to the new shared space
        src_vecs._matrix = np.dot(src_vecs._matrix, W)

        # Import datasets (representation will depend on final classifier)
        print('importing datasets')
        binary_dataset = General_Dataset(os.path.join('datasets', 'en', args.dataset), src_vecs,
                                         binary=True, rep=ave_vecs, one_hot=False, lowercase=False)
        binary_cross_dataset = General_Dataset(os.path.join('datasets', lang, args.dataset), trg_vecs,
                                               binary=True, rep=ave_vecs, one_hot=False, lowercase=False)

        fine_dataset = General_Dataset(os.path.join('datasets', 'en', args.dataset), src_vecs,
                                       binary=False, rep=ave_vecs, one_hot=False, lowercase=False)
        fine_cross_dataset = General_Dataset(os.path.join('datasets', lang, args.dataset), trg_vecs,
                                             binary=False, rep=ave_vecs, one_hot=False, lowercase=False)

        # Train linear SVM classifier
        if True in args.bi:
            best_c, best_f1 = get_best_C(binary_dataset, binary_cross_dataset)
            clf = LinearSVC(C=best_c)
            clf.fit(binary_dataset._Xtrain, binary_dataset._ytrain)
            cpred = clf.predict(binary_cross_dataset._Xtest)
            cf1 = macro_f1(binary_cross_dataset._ytest, cpred)
            print_prediction(clf, binary_cross_dataset,
                             os.path.join('predictions', lang, 'artetxe', '{0}-bi.txt'.format(args.dataset)))
            print('-binary-')
            print('Acc: {0:.3f}'.format(clf.score(binary_cross_dataset._Xtest, binary_cross_dataset._ytest)))
            print('Macro F1: {0:.3f}'.format(cf1))
            print()
            acc, prec, rec, f1 = scores(clf, binary_cross_dataset)
            print('acc:   {0:.3f}'.format(acc))
            print('prec:  {0:.3f}'.format(prec))
            print('rec:   {0:.3f}'.format(rec))
            print('f1:    {0:.3f}'.format(f1))

        if False in args.bi:
            best_c, best_f1 = get_best_C(fine_dataset, fine_cross_dataset)
            clf = LinearSVC(C=best_c)
            clf.fit(fine_dataset._Xtrain, fine_dataset._ytrain)
            cpred = clf.predict(fine_cross_dataset._Xtest)
            cf1 = macro_f1(fine_cross_dataset._ytest, cpred)
            print_prediction(clf, fine_cross_dataset,
                             os.path.join('predictions', lang, 'artetxe', '{0}-4cls.txt'.format(args.dataset)))
            print('-fine-')
            print('Acc: {0:.3f}'.format(clf.score(fine_cross_dataset._Xtest, fine_cross_dataset._ytest)))
            print('Macro F1: {0:.3f}'.format(cf1))
            print()
            acc, prec, rec, f1 = scores(clf, fine_cross_dataset)
            print('acc:   {0:.3f}'.format(acc))
            print('prec:  {0:.3f}'.format(prec))
            print('rec:   {0:.3f}'.format(rec))
            print('f1:    {0:.3f}'.format(f1))


if __name__ == '__main__':
    main()
