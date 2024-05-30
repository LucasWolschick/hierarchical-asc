from pathlib import Path
from typing import Any
import typing
import numpy as np
import pandas as pd
from dataclasses import dataclass

FileId = str


@dataclass
class Dataset:
    id: str

    root_path: Path

    id_from_path: dict[Path, FileId]
    path_from_id: dict[FileId, Path]
    labels: dict[FileId, str]

    train_names: list[FileId]
    test_names: list[FileId]
    eval_names: list[FileId]

    all_names: list[FileId]

    def frac(self, k: float):
        assert 0 <= k <= 1, f"fraction {k=} should be between 0 and 1"
        pool = list(self.all_names)
        n = int(len(pool) * k)
        gen = np.random.default_rng(42)
        all_names = gen.choice(pool, n, replace=False).tolist()
        d = set(all_names)
        train_names = [name for name in self.train_names if name in d]
        test_names = [name for name in self.test_names if name in d]
        eval_names = [name for name in self.eval_names if name in d]
        labels = {name: label for name, label in self.labels.items() if name in d}
        id_from_path = {path: id for path, id in self.id_from_path.items() if id in d}
        path_from_id = {id: path for id, path in self.path_from_id.items() if id in d}
        root_path = self.root_path
        id = f"{self.id}-frac{int(k*100)}"
        return Dataset(
            id=id,
            root_path=root_path,
            id_from_path=id_from_path,
            path_from_id=path_from_id,
            labels=labels,
            train_names=train_names,
            test_names=test_names,
            eval_names=eval_names,
            all_names=all_names,
        )


@typing.no_type_check
def tau2019dev():
    root = Path("datasets/tau2019/TAU-urban-acoustic-scenes-2019-development/")

    meta: Any = pd.read_csv(root / "meta.csv", sep="	")

    # 'filename' column is the id
    ids = meta["filename"].tolist()
    id_from_path = {root / id: id for id in ids}
    path_from_id = {v: k for k, v in id_from_path.items()}
    labels = {id: label for id, label in zip(ids, meta["scene_label"])}

    train_fold = pd.read_csv(root / "evaluation_setup/fold1_train.csv", sep="\t")
    test_fold = pd.read_csv(root / "evaluation_setup/fold1_test.csv", sep="\t")
    eval_fold = pd.read_csv(root / "evaluation_setup/fold1_evaluate.csv", sep="\t")

    print(train_fold.columns)
    train_names = train_fold["filename"].tolist()
    test_names = test_fold["filename"].tolist()
    eval_names = eval_fold["filename"].tolist()

    all_names = list(id_from_path.values())

    return Dataset(
        id="tau2019dev",
        root_path=root,
        id_from_path=id_from_path,
        path_from_id=path_from_id,
        labels=labels,
        train_names=train_names,
        test_names=test_names,
        eval_names=eval_names,
        all_names=all_names,
    )
