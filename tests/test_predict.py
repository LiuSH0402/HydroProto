from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydroproto.data import LABELS
from hydroproto.predict import (
    build_prediction_frame,
    load_prediction_csv,
    normalize_model_type,
)


def test_load_prediction_csv_without_labels(tmp_path):
    path = tmp_path / "sequences.csv"
    path.write_text(
        "accession,sequence\nA1,acde fg\nA2,MNPQ\n",
        encoding="utf-8",
    )
    frame = load_prediction_csv(path)
    assert frame.columns.tolist() == ["accession", "sequence"]
    assert frame["sequence"].tolist() == ["ACDEFG", "MNPQ"]


def test_load_prediction_csv_accepts_na_as_a_sequence(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("accession,sequence\nshort,NA\n", encoding="utf-8")
    frame = load_prediction_csv(path)
    assert frame.loc[0, "sequence"] == "NA"


def test_load_prediction_csv_rejects_duplicate_accessions(tmp_path):
    path = tmp_path / "duplicates.csv"
    path.write_text(
        "accession,sequence\nA1,ACDE\nA1,MNPQ\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate accessions"):
        load_prediction_csv(path)


def test_load_prediction_csv_rejects_invalid_amino_acids(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("accession,sequence\nA1,ACD*\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid amino-acid symbols"):
        load_prediction_csv(path)


def test_build_prediction_frame():
    frame = pd.DataFrame(
        {"accession": ["A1", "A2"], "sequence": ["ACDE", "MNPQRS"]}
    )
    probabilities = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.05, 0.05],
            [0.05, 0.1, 0.65, 0.1, 0.1],
        ]
    )
    output = build_prediction_frame(frame, probabilities, [1, 2])
    assert output["predicted_label"].tolist() == [LABELS[0], LABELS[2]]
    assert output["confidence"].tolist() == pytest.approx([0.7, 0.65])
    assert output["sequence_length"].tolist() == [4, 6]
    assert output["chunks"].tolist() == [1, 2]
    assert len([column for column in output if column.startswith("p_")]) == 5


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("hproto", "hydroproto"),
        ("proto", "hydroproto"),
        ("hlocal", "hydrolocal"),
        ("esm2-mlp", "mlp"),
        ("mpc", "hydrompc"),
    ],
)
def test_normalize_model_type(alias, expected):
    assert normalize_model_type(alias) == expected
