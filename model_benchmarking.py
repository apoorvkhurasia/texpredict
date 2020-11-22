import os
import shutil
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split

import feature_extraction as fe
import plothelp
from feature_extraction import FeatureExtractor
from models import KNNLearningModel, MLPLearningModel


class Benchmark:
    """
    Contains methods to run standardized benchmark tests on various symbol
    recognition models.
    """

    @staticmethod
    def run_benchmarks(training_data_dir: str, results_dir: str):
        """
        Runs benchmark tests and records observations.
        :param training_data_dir:   The directory which contains the training
                                    grid files and labels.
        :param results_dir:         The directory into which results should be
                                    written. The results are recorded in a file
                                    named `benchmark.csv` with each model's MCC,
                                    training time, and prediction time recorded.
                                    The method will also produce confusion
                                    matrices for each model benchmarked for
                                    debugging.
        """
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
            os.makedirs(results_dir, exist_ok=True)

        training_pdf = os.path.join(training_data_dir, "training_data.pdf")
        labels_file = os.path.join(training_data_dir,
                                   "training_labels.txt")
        conf_mat_figsize = (14, 14)
        if not os.path.isfile(training_pdf):
            raise IOError("Training input PDF not found.")
        if not os.path.isfile(labels_file):
            raise IOError("Training labels file not found.")

        training_file = fe.TrainingFileSpec(
                training_pdf, fe.DEFAULT_GRID_SPEC,
                labels_file, dpi=96, white_threshold=fe.DEFAULT_WHITE_THRESHOLD)

        angles = np.arange(-5, 5, 1)
        feature_vectors, feature_labels = \
            FeatureExtractor.extract_labelled_feature_vectors(training_file,
                                                              rotations=angles)

        train_ft, test_ft, train_lb, test_lb = train_test_split(feature_vectors,
                                                                feature_labels,
                                                                test_size=0.33,
                                                                random_state=81)
        all_labels = [lbl for lbl in np.unique(feature_labels)]

        # We still store our benchmarks here
        bench = pd.DataFrame(columns=['Model', 'MCC', 'TestTime', 'PredTime'])
        models = []

        # 1 hidden layer with various sizes
        for i in [26, 52, 80, 100]:
            models.append(MLPLearningModel(name="MLP (%i)" % i,
                                           hidden_layer_sizes=(i,),
                                           max_iter=1000))

        # 2 hidden layers with various sizes
        for i in [10, 26, 100]:
            for j in [10, 26, 100]:
                models.append(MLPLearningModel(name="MLP (%i, %i)" % (i, j),
                                               hidden_layer_sizes=(i, j,),
                                               max_iter=1000))

        # KNN models with various k values
        for k in [1, 5, 10, 15, 30]:
            models.append(KNNLearningModel(name="KNN (k=%i)" % k, k=k))

        for model in models:
            begin = time.time()
            model.train(train_ft, train_lb)
            test_time = time.time() - begin

            begin = time.time()
            predictions = model.predict(test_ft)
            pred_time = time.time() - begin

            mcc = matthews_corrcoef(y_pred=predictions, y_true=test_lb)
            bench = bench.append({'Model': model.name,
                                  'MCC': mcc,
                                  'TestTime': test_time,
                                  'PredTime': pred_time}, ignore_index=True)
            conf_mat = confusion_matrix(y_pred=predictions, y_true=test_lb,
                                        normalize='true')

            plot_title = "Confusion matrix for %s" % model.name
            fig = plothelp.plot_confusion_matrix(data=conf_mat,
                                                 title=plot_title,
                                                 figsize=conf_mat_figsize,
                                                 dpi=120,
                                                 labels=all_labels)
            plot_file_name = model.name.replace(" ", "_").replace(",", "_").\
                replace("(", "_").replace(")", "_").replace("=", "_") + ".png"
            fig.savefig(os.path.join(results_dir, plot_file_name))
            pyplot.close(fig)

        bench.to_csv(
                path_or_buf=os.path.join(results_dir, "benchmark.csv"),
                header=True, index=False)


if __name__ == '__main__':
    Benchmark.run_benchmarks(training_data_dir="data", results_dir="results")
    print("Finished benchmarking models. Please see results folder now.")
