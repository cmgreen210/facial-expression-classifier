from __future__ import print_function
import graphlab as gl
import pandas as pd
import numpy as np
import sys
from fec.classifier.gl_nn import GraphLabNeuralNetBuilder
from fec.classifier.gl_classifier import GraphLabClassifierFromNetBuilder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, f1_score
from os.path import join as pjoin
import os.path as opath


if __name__ == '__main__':

    net = gl.deeplearning.NeuralNet(url=sys.argv[1])
    if not net.verify():
        print("Invalid neural net! Exiting....")

    output_dir = sys.argv[2]
    check_point_path = pjoin(output_dir, 'chkpt')

    data_path = sys.argv[3]
    max_iterations = int(sys.argv[4])

    df = pd.read_pickle(data_path)

    cond_happy = df['django_expression'] == 3
    cond_sad = df['django_expression'] == 4
    cond_surprise = df['django_expression'] == 5

    df = df[cond_happy | cond_sad | cond_surprise]

    x = np.array(df['pixels'].tolist())
    y = np.array(df['django_expression'].values)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=.8)

    net_builder = GraphLabNeuralNetBuilder()
    net_builder.layers = net.layers
    net_builder.verify()

    model = GraphLabClassifierFromNetBuilder(net_builder,
                                             chkpt_dir=check_point_path,
                                             max_iterations=max_iterations,
                                             train_frac=.8)
    model.fit(xtrain, ytrain)

    eval = model.evaluate(xtest, ytest, metric=['accuracy', 'confusion_matrix',
                                               'recall@1', 'recall@2'])

    ypred = np.array(model.predict(xtest))
    ytest = np.array(ytest)

    test_f1 = f1_score(ytest, ypred)
    test_precision = precision_score(ytest, ypred)

    result_file = open(pjoin(output_dir, 'results.txt'), 'w')

    write = lambda val: print(val, file=result_file)
    write('accuracy, {0:1.6f}'.format(eval['accuracy']))
    write('recall@1, {0:1.6f}'.format(eval['recall@1']))
    write('recall@2, {0:1.6f}'.format(eval['recall@2']))
    write('precision, {0:1.6f}'.format(test_precision))
    write('f1, {0:1.6f}'.format(test_f1))
    write('')
    write('target_label, predicted_label, count')
    for row in eval['confusion_matrix']:
        write('{0}, {1}, {2}'.format(
            row['target_label'],
            row['predicted_label'],
            row['count']
        ))
    result_file.close()