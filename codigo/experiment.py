from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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
        models = [
            train_model(self.dataset, feature_data) for feature_data in feature_datas
        ]

        print("[4/4] Avaliando modelo")
        evaluate_model(
            self.dataset, list(zip(models, feature_datas)), fusion_rule=self.rule
        )


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


def train_model(
    dataset: Dataset, feature_data: dict[FileId, list[float]]
) -> BaseEstimator:
    train_names = dataset.train_names
    labeled_train_data = [(feature_data[id], dataset.labels[id]) for id in train_names]
    features = [x[0] for x in labeled_train_data]
    labels = [x[1] for x in labeled_train_data]

    # make template classifier
    scaler = StandardScaler()

    svc = SVC(probability=True)
    svc.set_params(kernel="rbf", class_weight="balanced")

    pipe = Pipeline(steps=[("scaler", scaler), ("svc", svc)])

    best_params = {"svc__C": 10, "svc__gamma": "auto"}
    pipe.set_params(**best_params)

    pipe.fit(features, labels)

    return pipe


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
    dataset: Dataset,
    features_models: list[tuple[BaseEstimator, dict[FileId, list[float]]]],
    fusion_rule: Any,
) -> Any:
    # if len(features_models) == 1:
    #     # no late fusion[
    #     model, features = features_models[0]

    #     eval_names = dataset.eval_names
    #     labeled_eval_data = [(features[id], dataset.labels[id]) for id in eval_names]
    #     feats = [x[0] for x in labeled_eval_data]
    #     labels = [x[1] for x in labeled_eval_data]

    #     print(
    #         "Acurácia/Precisão/Recall/F1-Score no conjunto de avaliação:",
    #         score_model(
    #             labels,
    #             model.predict(feats),
    #         ),
    #     )
    # else:

    # late fusion...
    # rule: sum probabilities and take argmax
    eval_names = dataset.eval_names
    labels = [dataset.labels[id] for id in eval_names]

    probas = []
    for model, features in features_models:
        feats = [features[id] for id in eval_names]
        proba = model.predict_proba(feats)
        probas.append(proba)

    final_proba = fusion_rule(probas)
    final_pred = np.argmax(final_proba, axis=1)
    final_pred = [model.classes_[p] for p in final_pred]

    print(
        "Acurácia/Precisão/Recall/F1-Score no conjunto de avaliação:",
        score_model(
            labels,
            final_pred,
        ),
    )