from openplaque.artery_detection import detect_artery_series

class DummyStudy:
    series = [
        {"SeriesNumber": 1035, "SeriesDescription": "RCA curved reformat coronary CPR"},
        {"SeriesNumber": 1039, "SeriesDescription": "CX curved reformat coronary CPR"},
        {"SeriesNumber": 1043, "SeriesDescription": "LAD curved reformat coronary CPR"},
    ]

def test_detect_artery_series():
    assert detect_artery_series(DummyStudy()) == {"LAD": 1043, "RCA": 1035, "LCX": 1039}
