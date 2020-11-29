import os
import shutil
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL.Image import Image
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import feature_extraction as fe
import imhelp
from feature_extraction import FeatureExtractor


class TeXLearningModel(ABC):

    @abstractmethod
    def train(self, training_features: np.ndarray, training_labels: List[str]):
        """Trains this model on given features and labels.

        :param training_features: An nxm array of features with one feature on
        each row and all pixels on columns.
        :param  training_labels: A list of n labels, one for each vector.
        """
        pass

    @abstractmethod
    def improve(self, training_features: np.ndarray, true_labels: List[str]):
        """
        Improves this model incrementally using the given features and labels.

        :param training_features: An nxm array of features with one feature on
        each row and all pixels on columns.
        :param  true_labels: A list of n labels, one for each vector.
        """
        pass

    @abstractmethod
    def predict(self, test_features: np.ndarray) -> List[str]:
        """Predicts labels for given feature vectors.

        :param test_features: An nxm array of features with one feature on
        each row and all pixels on columns.
        :returns: A list of n labels ---one for each vector.
        """
        pass

    def predict_equation(self, image: Image) -> str:
        """Given an image predicts the LaTeX equation for that image.

        :param image: An image.
        :returns: The predicted LaTeX equation for the image.
        """
        feature_vectors = []
        image = image.convert('L')
        while image is not None:
            image = imhelp.remove_border_and_padding(
                    image, black_threshold=0,
                    white_threshold=fe.DEFAULT_WHITE_THRESHOLD,
                    rem_black=True, rem_white=True)
            left_im, image = imhelp.extract_leftmost_symbol(image)
            fv = FeatureExtractor.extract_features_from_image(left_im)
            feature_vectors.append(fv)

        feature_matrix = fe.FeatureExtractor.get_feature_matrix(feature_vectors)
        return ' '.join(self.predict(feature_matrix))

    @abstractmethod
    def dump(self, state_dir: str):
        """Saves the state of this model to the given directory.

        :param state_dir: The directory.
        """
        pass

    @abstractmethod
    def load(self, state_dir: str):
        """Loads the state of this model from the given directory.

        :param state_dir: The directory.
        """
        pass


class KNNLearningModel(TeXLearningModel):
    def __init__(self, name: str, k: int):
        self.name = name
        self.model = KNeighborsClassifier(n_neighbors=k,
                                          metric='minkowski', p=2)
        self.scaler = None
        self.is_trained = False
        self.known_labels = []

    def train(self, training_features: np.ndarray, training_labels: List[str]):
        self.scaler = StandardScaler()
        self.scaler.fit(training_features)
        training_features = self.scaler.transform(training_features)
        self.model.fit(training_features, training_labels)
        self.known_labels.extend(training_labels)
        self.known_labels = [lbl for lbl in np.unique(self.known_labels)]
        self.is_trained = True

    def improve(self, training_features: np.ndarray, true_labels: List[str]):
        raise NotImplemented("Not implemented currently.")

    def predict(self, test_features: np.ndarray) -> List[str]:
        if not self.is_trained:
            raise Exception('train method must be called prior to '
                            'invoking this method')
        test_features = self.scaler.transform(test_features)
        return self.model.predict(test_features)

    def dump(self, state_dir: str):
        if os.path.isdir(state_dir):
            shutil.rmtree(state_dir)

        os.makedirs(state_dir, exist_ok=True)
        model_file = os.path.join(state_dir, "model.joblib")
        scaler_file = os.path.join(state_dir, "scaler.joblib")
        dump(self.model, model_file)
        dump(self.scaler, scaler_file)

    def load(self, state_dir: str):
        model_file = os.path.join(state_dir, "model.joblib")
        scaler_file = os.path.join(state_dir, "scaler.joblib")
        if not os.path.isfile(model_file):
            raise IOError("Invalid state. File %s not found." % model_file)
        if not os.path.isfile(scaler_file):
            raise IOError("Invalid state. File %s not found." % scaler_file)

        self.model = load(model_file)
        self.scaler = load(scaler_file)
        self.is_trained = True
        self.known_labels = self.model.classes_


class MLPLearningModel(TeXLearningModel):
    def __init__(self, name: str, max_iter=200, hidden_layer_sizes=(100,)):
        self.name = name
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                 max_iter=max_iter, activation='logistic',
                                 momentum=0.9)
        self.scaler = None
        self.is_trained = False
        self.known_labels = []

    def train(self, training_features: np.ndarray, training_labels: List[str]):
        self.scaler = StandardScaler()
        self.scaler.fit(training_features)
        training_features = self.scaler.transform(training_features)
        self.mlp.fit(training_features, training_labels)
        self.known_labels.extend(training_labels)
        self.known_labels = [lbl for lbl in np.unique(self.known_labels)]
        self.is_trained = True

    def improve(self, training_features: np.ndarray, true_labels: List[str]):
        if not self.is_trained:
            raise Exception('train method must be called prior to '
                            'invoking this method')
        training_features = self.scaler.transform(training_features)
        self.known_labels.extend(true_labels)
        self.known_labels = [lbl for lbl in np.unique(self.known_labels)]
        self.mlp.partial_fit(training_features, true_labels,
                             classes=self.known_labels)

    def predict(self, test_features: np.ndarray) -> List[str]:
        if not self.is_trained:
            raise Exception('train method must be called prior to '
                            'invoking this method')
        test_features = self.scaler.transform(test_features)
        return self.mlp.predict(test_features)

    def dump(self, state_dir: str):
        if os.path.isdir(state_dir):
            shutil.rmtree(state_dir)

        os.makedirs(state_dir, exist_ok=True)
        model_file = os.path.join(state_dir, "model.joblib")
        scaler_file = os.path.join(state_dir, "scaler.joblib")
        dump(self.mlp, model_file)
        dump(self.scaler, scaler_file)

    def load(self, state_dir: str):
        model_file = os.path.join(state_dir, "model.joblib")
        scaler_file = os.path.join(state_dir, "scaler.joblib")
        if not os.path.isfile(model_file):
            raise IOError("Invalid state. File %s not found." % model_file)
        if not os.path.isfile(scaler_file):
            raise IOError("Invalid state. File %s not found." % scaler_file)

        self.mlp = load(model_file)
        self.scaler = load(scaler_file)
        self.is_trained = True
        self.known_labels = [lbl for lbl in np.unique(self.mlp.classes_)]
