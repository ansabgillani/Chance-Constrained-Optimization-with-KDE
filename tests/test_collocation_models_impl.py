import numpy as np
from kde_cco.problems.lunar_landing import make_lunar_landing
from kde_cco.solvers.collocation import transcribe_euler

def test_euler_transcription_dimensions_and_defects():
    p = make_lunar_landing()
    t = np.linspace(0., p.tf, 5)
    X = np.zeros((5,2)); U=np.zeros(5)
    tr = transcribe_euler(p, t, X, U)
    assert tr['defects'].shape == (4,2)
    assert np.allclose(tr['defects'][:,0], 0.)
