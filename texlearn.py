import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import feature_extraction as fe
from feature_extraction import FeatureExtractor
from models import MLPLearningModel

if __name__ == "__main__":
    home = str(Path.home())
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("symbols_file",
                        help="Path of a PDF file which has grids "
                             "containing symbol images on each "
                             "page. Each page must contain images "
                             "of one symbol only and only one "
                             "symbol per grid.",
                        type=str)
    parser.add_argument("labels_file",
                        help="Path to a symbols file which is a "
                             "plain text file with LaTeX syntax "
                             "for a symbol on each line in the "
                             "order in which that symbol appears "
                             "in the symbols file.",
                        type=str)
    parser.add_argument("--state_dir",
                        help="A directory where model states will be stored.",
                        default="home", type=str)
    parser.add_argument("--reset", help="Forgets old training data.",
                        action="store_true")
    args = parser.parse_args()

    state_dir = os.path.join(
            home if args.state_dir == "home" else args.state_dir, ".texpredict")
    mlp_model = MLPLearningModel(name="default",
                                 hidden_layer_sizes=(80, ), max_iter=500)
    training_pdf = args.symbols_file
    labels_file = args.labels_file

    if args.reset and os.path.isdir(state_dir):
        shutil.rmtree(state_dir)

    if not os.path.isfile(training_pdf):
        raise IOError("Symbols file %s not found." % training_pdf)

    if not os.path.isfile(labels_file):
        raise IOError("Labels file %s not found." % labels_file)

    training_file = fe.TrainingFileSpec(
            training_pdf, fe.DEFAULT_GRID_SPEC,
            labels_file, dpi=96, white_threshold=fe.DEFAULT_WHITE_THRESHOLD)
    feature_matrix, feature_labels = \
        FeatureExtractor.extract_labelled_feature_vectors(
                training_file, rotations=np.zeros(1))

    if not os.path.isdir(state_dir):
        mlp_model.train(feature_matrix, feature_labels)
        mlp_model.dump(state_dir)
        print("Model trained. Program finished.")
    else:
        mlp_model.load(state_dir)
        mlp_model.improve(feature_matrix, feature_labels)
        mlp_model.dump(state_dir)
        print("Model improved. Program finished.")
