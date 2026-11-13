import numpy as np

from ecg import data as D


def test_inter_patient_split_has_no_shared_records():
    assert set(D.DS1_TRAIN).isdisjoint(D.DS2_TEST)
    assert len(D.DS1_TRAIN) == 22 and len(D.DS2_TEST) == 22
    assert not {102, 104, 107, 217} & set(D.DS1_TRAIN + D.DS2_TEST), "paced records must be excluded"


def test_split_by_record_keeps_patients_apart():
    data = D.make_synthetic(D.DS1_TRAIN + D.DS2_TEST, 20, 200, seed=0)
    tr, te = D.split_by_record(data, D.DS2_TEST)
    assert set(data.record[tr]).isdisjoint(set(data.record[te]))
    assert len(tr) + len(te) == len(data.y)


def test_split_by_beat_leaks_patients_on_purpose():
    data = D.make_synthetic(D.DS1_TRAIN + D.DS2_TEST, 20, 200, seed=0)
    tr, te = D.split_by_beat(data, 0.2, seed=0)
    assert len(set(data.record[tr]) & set(data.record[te])) > 0


def test_extract_beats_windows_labels_and_edges():
    signal = np.sin(np.linspace(0, 20, 2000))
    peaks = np.array([5, 500, 1000, 1995])          # first and last are too close to the edges
    symbols = ["N", "V", "+", "A"]                  # "+" is a rhythm marker, not a beat
    X, y = D.extract_beats(signal, peaks, symbols, pre=90, post=110)
    assert X.shape == (1, 200) and y.tolist() == [2]  # only the V beat at 500 survives
    assert abs(X[0].mean()) < 1e-5 and abs(X[0].std() - 1) < 1e-3  # z-scored


def test_symbol_mapping_covers_aami_classes():
    assert set(D.SYMBOL_TO_CLASS.values()) == set(range(len(D.AAMI_CLASSES)))
    assert D.SYMBOL_TO_CLASS["L"] == 0 and D.SYMBOL_TO_CLASS["A"] == 1 and D.SYMBOL_TO_CLASS["V"] == 2


def test_validation_records_come_from_training_records():
    val = D.validation_records(D.DS1_TRAIN, 4, seed=0)
    assert len(val) == 4 and set(val) <= set(D.DS1_TRAIN)
    assert val == D.validation_records(D.DS1_TRAIN, 4, seed=0)  # reproducible
