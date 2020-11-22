import os

import numpy as np
from pdf2image import convert_from_path

import feature_extraction as fe
from feature_extraction import FeatureExtractor
from models import MLPLearningModel

if __name__ == "__main__":
    state_dir = os.path.join("state", "demo")
    mlp_model = MLPLearningModel(name="default",
                                 hidden_layer_sizes=(80, ), max_iter=500)
    if not os.path.isdir(state_dir):
        training_pdf = os.path.join("data", "training_data.pdf")
        labels_file = os.path.join("data", "training_labels.txt")
        training_file = fe.TrainingFileSpec(
                training_pdf, fe.DEFAULT_GRID_SPEC,
                labels_file, dpi=96, white_threshold=fe.DEFAULT_WHITE_THRESHOLD)
        feature_matrix, feature_labels = \
            FeatureExtractor.extract_labelled_feature_vectors(
                    training_file, rotations=np.zeros(1))

        mlp_model.train(feature_matrix, feature_labels)
        mlp_model.dump(state_dir)
    else:
        mlp_model.load(state_dir)

    pages = convert_from_path("demo/testing_data.pdf", dpi=96,
                              transparent=False, grayscale=True)
    for index, page in enumerate(pages):
        print("Page %i: %s" % (index, mlp_model.predict_equation(page)))
