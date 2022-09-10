from mlproject.computedist import haversine

def test_haversine():
    assert round(haversine(48.865070,2.380009,48.235070,2.393409),0) == 70
