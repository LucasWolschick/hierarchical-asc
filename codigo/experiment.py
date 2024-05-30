from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import concurrent.futures
import tqdm
import matplotlib.pyplot as plt

from dataset import Dataset, FileId
from features.feature_vector import FeatureSet
import spectrogram as spectrogram


@dataclass
class Experiment:
    dataset: Dataset
    feature_sets: list[FeatureSet]
    rule: Any

    def run(self):
        print("[1/4] Gerando espectrogramas")
        spectrograms = prepare_spectrograms(self.dataset)

        print("[2/4] Extraindo características")
        feature_datas = [
            extract_features(self.dataset, set, spectrograms)
            for set in self.feature_sets
        ]

        print("[3/4] Treinando modelo")
        train_names = self.dataset.train_names
        train_labels = [self.dataset.labels[id] for id in train_names]
        train_datas = [
            select_features(train_names, feature_data) for feature_data in feature_datas
        ]
        model = train_model(train_datas, train_labels, self.rule)

        print("[4/4] Avaliando modelo")
        eval_names = self.dataset.eval_names
        eval_labels = [self.dataset.labels[id] for id in eval_names]
        eval_datas = [
            select_features(eval_names, feature_data) for feature_data in feature_datas
        ]
        evaluate_model(model, eval_datas, eval_labels)


def prepare_spectrograms(dataset: Dataset) -> dict[FileId, Path]:
    spectrogram_base = Path(dataset.root_path) / "spectrograms"
    spectrogram_base.mkdir(parents=True, exist_ok=True)

    def process_file(id: FileId):
        audio_path = dataset.path_from_id[id]
        img_path = spectrogram_base / (audio_path.stem + ".png")
        if not img_path.exists():
            s = spectrogram.generate_spectrogram(audio_path)
            spectrogram.save_spectrogram_image_to_path(s, img_path)
        return (id, img_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        return dict(
            tqdm.tqdm(
                executor.map(process_file, dataset.all_names),
                total=len(dataset.all_names),
            )
        )


def extract_features(
    dataset: Dataset, feature_set: FeatureSet, spectrograms: dict[FileId, Path]
) -> dict[FileId, list[float]]:
    print(f"Extraindo features ({id})")

    # extrai feature
    features = dataset.root_path / f"feature_data_{dataset.id}_{feature_set.id}.pkl"
    if features.exists():
        with open(features, "rb") as f:
            p = pickle.load(f)
        return p

    def process_file(f: tuple[FileId, Path]) -> tuple[FileId, list[float]]:
        id: FileId = f[0]
        spectrogram_path: Path = f[1]
        feature_vector = feature_set.extract(dataset.path_from_id[id], spectrogram_path)
        return (id, feature_vector)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        feature_data: list[tuple[FileId, list[float]]] = list(
            tqdm.tqdm(
                executor.map(process_file, spectrograms.items()),
                total=len(spectrograms),
            )
        )  # type: ignore

    data = dict(feature_data)
    with open(features, "wb") as f:
        pickle.dump(data, f)

    return data


def select_features(
    keys: list[FileId], features: dict[FileId, list[float]]
) -> list[list[float]]:
    return [features[id] for id in keys]


def train_model(Xs: list[list[list[float]]], y: list[str], rule=lambda x: x[0]):
    # template classifier
    scaler = StandardScaler()
    svc = SVC(
        probability=True, kernel="rbf", class_weight="balanced", C=10, gamma="auto"
    )
    pipe = Pipeline(steps=[("scaler", scaler), ("svc", svc)])

    # late fusion
    lfe = LateFusionEstimator(base_estimator=pipe, fusion_rule=rule)

    lfe.fit(Xs, y)

    return lfe


def score_model(y, y_pred):
    # confusion matrix
    cm = ConfusionMatrixDisplay.from_predictions(y, y_pred, xticks_rotation=45)
    plt.show()

    return (
        accuracy_score(y, y_pred),
        precision_score(y, y_pred, average="weighted"),
        recall_score(y, y_pred, average="weighted"),
        f1_score(y, y_pred, average="weighted"),
    )


def evaluate_model(
    model,
    Xs: list[list[list[float]]],
    y: list[str],
) -> Any:
    final_pred = model.predict(Xs)

    print(
        "Acurácia/Precisão/Recall/F1-Score no conjunto de avaliação:",
        score_model(
            y,
            final_pred,
        ),
    )


class LateFusionEstimator:
    def __init__(self, base_estimator, fusion_rule=lambda x: x[0]):
        self.base_estimator = base_estimator
        self.rule = fusion_rule
        self.models_ = []

    def fit(self, Xs: list[list[list[float]]], y: list[str]):
        for X in Xs:
            model = clone(self.base_estimator)
            model.fit(X, y)
            self.models_.append(model)
        return self

    def predict(self, Xs):
        assert len(Xs) == len(
            self.models_
        ), "Este modelo não foi treinado para esse dataset."

        final_proba = self.predict_proba(Xs)
        final_pred = np.argmax(final_proba, axis=1)
        return self.models_[0].classes_[final_pred]

    def predict_proba(self, Xs):
        probas = [model.predict_proba(X) for model, X in zip(self.models_, Xs)]
        return self.rule(probas)
