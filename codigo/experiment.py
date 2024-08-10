from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator
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
import hiclass

import hierarchy as hier
from dataset import Dataset, FileId
from features.feature_vector import FeatureSet
import spectrogram as spectrogram


@dataclass
class Experiment:
    dataset: Dataset
    feature_sets: list[FeatureSet]
    rule: Any
    log: bool = True

    def print(self, *args, **kwargs):
        if self.log:
            print(*args, **kwargs)

    def run(self):
        self.print("[1/4] Gerando espectrogramas")
        spectrograms = prepare_spectrograms(self.dataset, self.log)

        self.print("[2/4] Extraindo características")
        feature_datas = [
            extract_features(self.dataset, set, spectrograms, self.log)
            for set in self.feature_sets
        ]

        self.print("[3/4] Treinando modelo")
        train_names = self.dataset.train_names
        train_labels = [self.dataset.labels[id] for id in train_names]
        train_datas = [
            select_features(train_names, feature_data) for feature_data in feature_datas
        ]
        model = train_model(train_datas, train_labels, self.rule)

        self.print("[4/4] Avaliando modelo")
        eval_names = self.dataset.eval_names
        eval_labels = [self.dataset.labels[id] for id in eval_names]
        eval_datas = [
            select_features(eval_names, feature_data) for feature_data in feature_datas
        ]
        return evaluate_model(model, eval_datas, eval_labels)

    def run_hier_inner(self, paths, classifier: Any = hiclass.LocalClassifierPerNode):
        self.print("[1/4] Gerando espectrogramas")
        spectrograms = prepare_spectrograms(self.dataset, self.log)

        self.print("[2/4] Extraindo características")
        feature_datas = [
            extract_features(self.dataset, set, spectrograms, self.log)
            for set in self.feature_sets
        ]

        self.print("[3/4] Treinando modelo")
        train_names = self.dataset.train_names
        train_labels = [paths[self.dataset.labels[id]] for id in train_names]
        train_datas = [
            select_features(train_names, feature_data) for feature_data in feature_datas
        ]
        model = train_hierarchical_model_inner(
            train_datas, train_labels, self.rule, classifier
        )

        self.print("[4/4] Avaliando modelo")
        eval_names = self.dataset.eval_names
        eval_labels = [paths[self.dataset.labels[id]] for id in eval_names]
        eval_datas = [
            select_features(eval_names, feature_data) for feature_data in feature_datas
        ]
        return evaluate_model(model, eval_datas, eval_labels)

    def run_hier_outer(self, paths, classifier: Any = hiclass.LocalClassifierPerNode):
        self.print("[1/4] Gerando espectrogramas")
        spectrograms = prepare_spectrograms(self.dataset, self.log)

        self.print("[2/4] Extraindo características")
        feature_datas = [
            extract_features(self.dataset, set, spectrograms)
            for set in self.feature_sets
        ]

        self.print("[3/4] Treinando modelo")
        train_names = self.dataset.train_names
        train_labels = [paths[self.dataset.labels[id]] for id in train_names]
        train_datas = [
            select_features(train_names, feature_data) for feature_data in feature_datas
        ]
        model = train_hierarchical_model_outer(
            train_datas, train_labels, self.rule, classifier
        )

        self.print("[4/4] Avaliando modelo")
        eval_names = self.dataset.eval_names
        eval_labels = [paths[self.dataset.labels[id]] for id in eval_names]
        eval_datas = [
            select_features(eval_names, feature_data) for feature_data in feature_datas
        ]
        return evaluate_model(model, eval_datas, eval_labels)


def prepare_spectrograms(dataset: Dataset, log: bool = True) -> dict[FileId, Path]:
    spectrogram_base = Path(dataset.root_path) / "spectrograms"
    spectrogram_base.mkdir(parents=True, exist_ok=True)

    def process_file(id: FileId):
        audio_path = dataset.path_from_id[id]
        img_path = spectrogram_base / (audio_path.stem + ".png")
        if not img_path.exists():
            s = spectrogram.generate_spectrogram(audio_path)
            spectrogram.save_spectrogram_image_to_path(s, img_path)
        return (id, img_path)

    if log:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return dict(
                tqdm.tqdm(
                    executor.map(process_file, dataset.all_names),
                    total=len(dataset.all_names),
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return dict(
                executor.map(process_file, dataset.all_names),
            )


def extract_features(
    dataset: Dataset,
    feature_set: FeatureSet,
    spectrograms: dict[FileId, Path],
    log: bool = True,
) -> dict[FileId, list[float]]:
    # print(f"Extraindo features ({id})")

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

    if log:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            feature_data: list[tuple[FileId, list[float]]] = list(
                tqdm.tqdm(
                    executor.map(process_file, spectrograms.items()),
                    total=len(spectrograms),
                )
            )  # type: ignore
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            feature_data: list[tuple[FileId, list[float]]] = list(
                executor.map(process_file, spectrograms.items()),
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
    def factory():
        scaler = StandardScaler()
        svc = SVC(
            probability=True, kernel="rbf", class_weight="balanced", C=10, gamma="auto"
        )
        pipe = Pipeline(steps=[("scaler", scaler), ("svc", svc)])
        return pipe

    # split
    X, ends = merge_features(Xs)

    # late fusion
    lfe = LateFusionEstimator(
        base_estimator_factory=factory, fusion_rule=rule, ends=ends
    )

    lfe.fit(X, y)

    return lfe


def merge_features(Xs: list[list[list[float]]]) -> tuple[list[list[float]], list[int]]:
    acc = 0
    ends = []
    all = []

    # join feature vectors
    for i in range(len(Xs[0])):
        lst = []
        for feat_list in Xs:
            lst += feat_list[i]
        all.append(lst)

    # calculate ends
    for x in Xs:
        acc += len(x[0])
        ends.append(acc)

    return all, ends


def unmerge_features(X: list[list[float]], ends: list[int]) -> list[list[list[float]]]:
    last = 0

    ranges = []
    for end in ends:
        ranges.append((last, end))
        last = end

    lsts = [[feat_vec[range[0] : range[1]] for feat_vec in X] for range in ranges]
    return lsts


def train_hierarchical_model_inner(
    Xs: list[list[list[float]]],
    y: list[str],
    rule=lambda x: x[0],
    classifier=hiclass.LocalClassifierPerNode,
):
    # MERGE
    merged, ends = merge_features(Xs)

    # template classifier
    def factory():
        scaler = StandardScaler()
        svc = SVC(
            probability=True, kernel="rbf", class_weight="balanced", C=10, gamma="auto"
        )
        pipe = Pipeline(steps=[("scaler", scaler), ("svc", svc)])
        return pipe

    # late fusion
    lfe = LateFusionEstimator(
        base_estimator_factory=factory, fusion_rule=rule, ends=ends
    )

    # hier
    return classifier(
        local_classifier=lfe,
        # n_jobs=os.cpu_count() or 1,
    ).fit(merged, y)


def train_hierarchical_model_outer(
    Xs: list[list[list[float]]],
    y: list[str],
    rule=lambda x: x[0],
    classifier=hiclass.LocalClassifierPerNode,
):
    # template classifier
    def factory():
        scaler = StandardScaler()
        svc = SVC(
            probability=True, kernel="rbf", class_weight="balanced", C=10, gamma="auto"
        )
        pipe = Pipeline(steps=[("scaler", scaler), ("svc", svc)])

        LCPN = classifier(
            local_classifier=pipe,
            # n_jobs=os.cpu_count() or 1,
        )
        return LCPN

    # merge
    X, ends = merge_features(Xs)

    # late fusion
    return LateFusionEstimator(
        base_estimator_factory=factory, fusion_rule=rule, ends=ends
    ).fit(X, y)


def score_model(y, y_pred):
    # confusion matrix
    # cm = ConfusionMatrixDisplay.from_predictions(y, y_pred, xticks_rotation=45)
    # plt.show()

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
    merged, _ = merge_features(Xs)

    final_pred = model.predict(merged)
    elems = y

    if isinstance(y[0], list) or isinstance(y[0], np.ndarray):
        elems = [elem[-1] for elem in y]

    if isinstance(final_pred[0], list) or isinstance(final_pred[0], np.ndarray):
        final_pred = [elem[-1] for elem in final_pred]

    s = score_model(
        elems,
        final_pred,
    )
    print("Acurácia/Precisão/Recall/F1-Score no conjunto de avaliação:", s)
    return s


class LateFusionEstimator(BaseEstimator):
    def __init__(self, base_estimator_factory, fusion_rule=lambda x: x[0], ends=None):
        self.base_estimator_factory = base_estimator_factory
        self.rule = fusion_rule
        self.ends = ends
        self.models_ = []

    def fit(self, X: list[list[float]], y: list[str], class_weight=None):
        if self.ends:
            split = unmerge_features(X, self.ends)
            for X in split:
                model = self.base_estimator_factory()
                model.fit(X, y)
                self.models_.append(model)
            self.classes_ = self.models_[0].classes_
        else:
            model = self.base_estimator_factory()
            model.fit(X, y)
            self.models_.append(model)
            self.classes_ = model.classes_
        return self

    def predict(self, X):
        final_proba = self.predict_proba(X)
        final_pred = np.argmax(final_proba, axis=1)
        clist = self.models_[0].classes_

        if not isinstance(clist[0], str):
            pred = np.array(clist[-1])[final_pred]
        else:
            pred = np.array(clist)[final_pred]
        return pred

    def predict_proba(self, X):
        if self.ends:
            Xs = unmerge_features(X, self.ends)
            probas = [model.predict_proba(X) for model, X in zip(self.models_, Xs)]
            return self.rule(probas)
        else:
            return self.models_[0].predict_proba(X)
