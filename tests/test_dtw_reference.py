"""Optional numerical comparison with the dtaidistance reference implementation."""
import unittest
import numpy as np

from src.runtime.pxrd_dtw import constrained_dtw_distance, prepare_profile_pair

try:
    from dtaidistance import dtw
except ImportError:
    dtw = None


@unittest.skipIf(dtw is None, "install the verification extra for reference DTW parity")
class DtwReferenceTest(unittest.TestCase):
    def test_matches_reference_python_and_c_implementations(self):
        rng = np.random.default_rng(20261008)
        for length in (7, 61, 137, 3400):
            for radius in (0, 1, 7, 30):
                with self.subTest(length=length, radius=radius):
                    a, b = prepare_profile_pair(rng.random(length), rng.random(length),
                                                pooling_factor=1 if length == 7 else 5)
                    actual = constrained_dtw_distance(a, b, radius)
                    for function in (dtw.distance, dtw.distance_fast):
                        expected = function(a, b, inner_dist="euclidean", window=radius + 1,
                                            penalty=0, psi=0, use_pruning=False) / np.sqrt(len(a))
                        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
