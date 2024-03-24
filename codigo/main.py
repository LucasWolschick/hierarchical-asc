# Este script realiza todos os passos do experimento


import os
from pathlib import Path
import concurrent.futures

import numpy as np

from joblib import cpu_count
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV
import spectrogram
import tqdm
from features.feature_vector import *
from hiclass import (
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
    LocalClassifierPerLevel,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEV_DS = Path.home() / "pic/datasets/tau2019/TAU-urban-acoustic-scenes-2019-development"

HIERARCHY = {
    # hierarchy 1
    "_root": ["_indoors", "_outdoors", "_transportation"],
    "_indoors": ["airport", "shopping_mall", "metro_station"],
    "_outdoors": ["park", "street_traffic", "street_pedestrian", "public_square"],
    "_transportation": ["tram", "bus", "metro"],
    # hierarchy 2
    # "_root": ["_indoors", "_outdoors", "_transportation"],
    # "_indoors": ["_airport_shopping", "metro_station"],
    # "_airport_shopping": ["airport", "shopping_mall"],
    # "_outdoors": ["park", "street_traffic", "_streets"],
    # "_streets": ["street_pedestrian", "public_square"],
    # "_transportation": ["_tram_bus", "metro"],
    # "_tram_bus": ["tram", "bus"],
    # hierarchy 3
    # "indoors": ["airport2", "shopping_mall2", "metro_station2"],
    # "outdoors": ["street_pedestrian2", "park2", "outdoors_park"],
    # "outdoors_park": ["public_square", "street_traffic"],
    # "transportation": ["tram2", "bus2", "metro2"],
    # "airport2": ["airport"],
    # "shopping_mall2": ["shopping_mall"],
    # "metro_station2": ["metro_station"],
    # "street_pedestrian2": ["street_pedestrian"],
    # "park2": ["park"],
    # "tram2": ["tram"],
    # "bus2": ["bus"],
    # "metro2": ["metro"],
}


def leaves(hierarchy, node):
    if node not in hierarchy:
        return [node]
    else:
        return sum([leaves(hierarchy, child) for child in hierarchy[node]], [])


def save_development_dataset_spectrograms():
    print("Gerando espectrogramas")

    (DEV_DS / "spectrograms").mkdir(exist_ok=True)

    df = pd.read_csv(DEV_DS / "meta.csv", sep="\t")

    filenames = df["filename"].values

    def process_file(filename):
        # print(f"processing {filename}")
        audio_path = DEV_DS / filename
        img_path = DEV_DS / "spectrograms" / (Path(filename).stem + ".png")
        if not img_path.exists():
            s = spectrogram.generate_spectrogram(audio_path)
            spectrogram.save_spectrogram_image_to_path(s, img_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(process_file, filenames), total=len(filenames)))

    # for i in tqdm.tqdm(range(len(filenames))):
    #     filename = filenames[i]
    #     audio_path = DEV_DS / filename
    #     img_path = DEV_DS / "spectrograms" / (Path(filename).stem + ".png")
    #     if not img_path.exists():
    #         s = spectrogram.generate_spectrogram(audio_path)
    #         spectrogram.save_spectrogram_image_to_path(s, img_path)


def extrai_feature_set(id, extrator):
    print(f"Extraindo features ({id})")

    # extrai feature
    feature_file = DEV_DS / f"feature_data_{id}.pkl"

    if os.path.exists(feature_file):
        return pd.read_pickle(feature_file)

    df = pd.read_csv(DEV_DS / "meta.csv", sep="\t")

    def process_file(filename):
        audio_path = DEV_DS / filename
        img_path = DEV_DS / "spectrograms" / (Path(filename).stem + ".png")
        feature_vector = extrator(audio_path, img_path)
        return feature_vector

    with concurrent.futures.ThreadPoolExecutor() as executor:
        feature_data = list(
            tqdm.tqdm(
                executor.map(process_file, df["filename"].to_list()),
                total=len(df["filename"]),
            )
        )

    # feature_data = []
    # for i in tqdm.tqdm(range(len(df))):
    #     filename = df["filename"][i]
    #     feature_vector = process_file(filename)
    #     feature_data.append(feature_vector)

    df["feature_vector"] = feature_data
    df.to_pickle(feature_file)
    return df


def extract_features():
    # feature_data_1 = extrai_feature_set(feature_vector_1_id(), feature_vector_1)
    # feature_data_2 = extrai_feature_set(feature_vector_2_id(), feature_vector_2)

    # feature_data = pd.merge(
    #     feature_data_1, feature_data_2, on="filename", suffixes=("_1", "_2")
    # )

    id, extrator = fv_lbp
    feature_data = extrai_feature_set(id, extrator)

    return feature_data


def train_model(feature_data):
    print("Treinando classificador")

    train_df = pd.read_csv(DEV_DS / "evaluation_setup/fold1_train.csv", sep="\t")
    train_data = feature_data[feature_data["filename"].isin(train_df["filename"])][
        ["feature_vector", "scene_label"]
    ]

    # make template classifier
    scaler = StandardScaler()

    svc = SVC()
    svc.set_params(kernel="rbf", class_weight="balanced")

    pipe = Pipeline(steps=[("scaler", scaler), ("svc", svc)])

    # grid search
    # param_grid = {
    #     "svc__C": np.linspace(10, 50, 5),
    #     "svc__gamma": np.logspace(-3, -1, 5),
    # }
    # search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=3)
    # search.fit(
    #     train_data["feature_vector"].tolist(),
    #     train_data["scene_label"].tolist(),
    # )
    # print("Melhores parâmetros:", search.best_params_)
    # pipe.set_params(**search.best_params_)
    best_params = {"svc__C": 10, "svc__gamma": "auto"}
    pipe.set_params(**best_params)

    pipe.fit(
        train_data["feature_vector"].tolist(),
        train_data["scene_label"].tolist(),
    )

    return pipe


def train_hierarchical_model(feature_data):
    print("Treinando classificador hierárquico LCPPN")

    def classe_pai(classe):
        for pai, filhos in HIERARCHY.items():
            if classe in filhos:
                return pai
        return classe

    train_df = pd.read_csv(DEV_DS / "evaluation_setup/fold1_train.csv", sep="\t")
    train_data = feature_data[feature_data["filename"].isin(train_df["filename"])][
        ["feature_vector", "scene_label"]
    ]

    # make template classifier
    scaler = StandardScaler()
    svc = SVC()
    svc.set_params(kernel="rbf", probability=True, class_weight="balanced")
    pipe = Pipeline(steps=[("scaler", scaler), ("svc", svc)])
    best_params = {"svc__C": 10, "svc__gamma": "auto"}
    pipe.set_params(**best_params)

    # grid search
    # param_grid = {
    #     "svc__C": [10**i for i in range(-3, 4)],
    #     "svc__gamma": [10**i for i in range(-3, 4)],
    # }
    # search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5, verbose=3)
    # search.fit(
    #     train_data["feature_vector"].tolist(),
    #     train_data["scene_label"].tolist(),
    # )
    # print("Melhores parâmetros:", search.best_params_)

    # pipe.set_params(**search.best_params_)

    # build cached hierarchy paths starting from root classes
    def path(label: str):
        parent = classe_pai(label)
        if parent == label:
            return [label]
        else:
            return path(parent) + [label]

    all_classes = set(sum(HIERARCHY.values(), []))
    all_classes = all_classes.union(HIERARCHY.keys())
    hierarchy_paths = {classe: path(classe) for classe in all_classes}

    train_X, train_y = (
        train_data["feature_vector"].tolist(),
        train_data["scene_label"].tolist(),
    )
    train_y = [hierarchy_paths[label] for label in train_y]
    print(train_y[:5])

    # make LCPN and fit it
    lcpn1 = LocalClassifierPerParentNode(local_classifier=pipe, n_jobs=cpu_count() + 1)
    lcpn1.fit(train_X, train_y)

    return lcpn1


def score_model(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X)
    if any(
        isinstance(model, x)
        for x in [
            LocalClassifierPerParentNode,
            LocalClassifierPerNode,
            LocalClassifierPerLevel,
        ]
    ):
        # flatten the predictions to the last element only
        y_pred = [p[-1] for p in y_pred]

    conf_matrix = confusion_matrix(y, y_pred, labels=leaves(HIERARCHY, "_root"))
    print(conf_matrix)
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, labels=leaves(HIERARCHY, "_root")
    ).plot()
    plt.savefig("out.png")

    return (
        accuracy_score(y, y_pred),
        precision_score(y, y_pred, average="weighted"),
        recall_score(y, y_pred, average="weighted"),
        f1_score(y, y_pred, average="weighted"),
    )


def score_model_late(classifiers, Xs, y):
    from sklearn.metrics import accuracy_score

    # preds is a list of predictions for each classifier,
    # where each prediction is a list of probabilities for each class
    preds = []
    for model, X in zip(classifiers, Xs):
        y_pred = model.predict_proba(X)
        preds.append(y_pred)

    # compute final probabilities for each class using product of probabilities in preds
    final_probs = []
    for i in range(len(preds[0])):
        prob = 1
        for j in range(len(classifiers)):
            prob *= preds[j][i]
        final_probs.append(prob)

    # get the class (name) with the highest probability
    y_pred = [classifiers[0].classes_[p.argmax()] for p in final_probs]

    return accuracy_score(y, y_pred)

    # if isinstance(model, LocalClassifierPerParentNode):
    #     # flatten the predictions to the last element only
    #     y_pred = [p[-1] for p in y_pred]


def evaluate_model(feature_data, model):
    print("Avaliando modelo...")
    # print(feature_data)
    eval_df = pd.read_csv(DEV_DS / "evaluation_setup/fold1_evaluate.csv", sep="\t")
    eval_data = feature_data[feature_data["filename"].isin(eval_df["filename"])][
        ["feature_vector", "scene_label"]
    ]
    train_df = pd.read_csv(DEV_DS / "evaluation_setup/fold1_train.csv", sep="\t")
    train_data = feature_data[feature_data["filename"].isin(train_df["filename"])][
        ["feature_vector", "scene_label"]
    ]
    # print(
    #     "Acurácia/Precisão/Recall/F1-Score no conjunto de treinamento:",
    #     score_model(
    #         model,
    #         train_data["feature_vector"].tolist(),
    #         train_data["scene_label"].tolist(),
    #     ),
    # )
    print(
        "Acurácia/Precisão/Recall/F1-Score no conjunto de avaliação:",
        score_model(
            model,
            eval_data["feature_vector"].tolist(),
            eval_data["scene_label"].tolist(),
        ),
    )
    # print(
    #     "Acurácia/Precisão/Recall/F1-Score no conjunto completo de dados:",
    #     score_model(
    #         model,
    #         feature_data["feature_vector"].tolist(),
    #         feature_data["scene_label"].tolist(),
    #     ),
    # )


if __name__ == "__main__":
    save_development_dataset_spectrograms()
    feature_data = extract_features()
    model = train_model(feature_data)
    # model = train_hierarchical_model(feature_data)
    evaluate_model(feature_data, model)
