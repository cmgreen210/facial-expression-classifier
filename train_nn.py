from __future__ import print_function
import graphlab as gl
import pandas as pd
import numpy as np
import sys
from fec.classifier.gl_nn import GraphLabNeuralNetBuilder
from fec.classifier.gl_classifier import *
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import precision_score, f1_score
from os.path import join as pjoin
import os
import StringIO
import shutil


def fix_net_conf(path):
    """Fix net layer connection numbers

    :param path: path to net config
    """
    f = open(path, 'r')
    layer_count = 0
    output = StringIO.StringIO()
    for line in f.readlines():
        if 'layer' in line:
            layer_line = line.split('=')
            layer = 'layer[' + str(layer_count) +\
                    "->" + str(layer_count+1) + ']'
            line = layer + " =" + layer_line[1]
            layer_count += 1
        output.write(line)
    f.close()
    f = open(path, 'w')
    f.write(output.getvalue())
    f.close()


def assemble_data_frame(path):
    df = pd.read_pickle(path)

    cond_happy = df['emotion'] == 3
    cond_sad = df['emotion'] == 4
    cond_surprise = df['emotion'] == 5

    return df[cond_happy | cond_sad | cond_surprise]


if __name__ == '__main__':
    net_conf_path = sys.argv[1]
    fix_net_conf(net_conf_path)

    net = gl.deeplearning.NeuralNet(url=net_conf_path)
    if not net.verify():
        print("Invalid neural net! Exiting....")

    output_dir = sys.argv[2]
    check_point_path = pjoin(output_dir, 'chkpt')

    data_path = sys.argv[3]
    max_iterations = int(sys.argv[4])

    cross_validation = None
    if len(sys.argv) > 5:
        if sys.argv[5] != '':
            cross_validation = int(sys.argv[5])

    df = assemble_data_frame(data_path)

    x = np.array(df['pixels'].tolist())
    y = np.array(df['emotion'].values)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                    train_size=.8,
                                                    random_state=1)

    net_builder = GraphLabNeuralNetBuilder()
    net_builder.layers = net.layers
    net_builder.verify()

    if cross_validation is None:

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
        final = os.path.join(check_point_path, 'final')
        os.mkdir(final)
        model.save(final)
    else:
        kf = KFold(xtrain.shape[0], n_folds=cross_validation)
        count = 1
        result_file = open(pjoin(output_dir, 'results.txt'), 'w')
        write = lambda val: print(val, file=result_file)
        for train, validation in kf:

            xtrain_kf = np.copy(xtrain)
            ytrain_kf = np.copy(ytrain)
            xtest_kf = np.copy(xtest)
            ytest_kf = np.copy(ytest)

            kf_chkpt = pjoin(check_point_path, "kv_" + str(count))
            if os.path.exists(kf_chkpt):
                shutil.rmtree(kf_chkpt)
            os.mkdir(kf_chkpt)

            kf_final = pjoin(check_point_path, "kv_" + str(count) + "_final")
            if os.path.exists(kf_final):
                shutil.rmtree(kf_final)
            os.mkdir(kf_final)

            model = GraphLabClassifierFromNetBuilder(
                net_builder,
                chkpt_dir=kf_chkpt,
                max_iterations=max_iterations,
                validation_set=(np.copy(xtrain_kf[validation]),
                                np.copy(ytrain_kf[validation])))

            model.fit(np.copy(xtrain_kf[train]), np.copy(ytrain_kf[train]))

            eval = model.evaluate(xtest_kf, ytest_kf,
                                  metric=['accuracy',
                                          'confusion_matrix',
                                          'recall@1',
                                          'recall@2'])

            ypred = np.array(model.predict(xtest_kf))
            ytest_arr = np.array(ytest_kf)

            test_f1 = f1_score(ytest_arr, ypred)
            test_precision = precision_score(ytest_arr, ypred)

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
            write('')
            count += 1

            model.save(kf_final)

        result_file.close()