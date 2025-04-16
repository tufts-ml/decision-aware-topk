import unittest
import torch  
import numpy as np
import pytest
from spatiotemporalforecaster.dataset import Dataset
import re

def valid_arrays():
    """Return a set of valid dummy arrays for the Dataset initializer."""
    T, S, F = 3, 5, 7
    dynamic_feats_TSF = np.zeros((T, S, F))
    static_feats_SF = np.zeros((S, F))
    temp_feats_TF = np.zeros((T, F))
    adj_SS = np.zeros((S, S))
    return dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS

class TestDataset(unittest.TestCase):

    """
    Test to_3D
    """



    
    """
    Test __init__
    """
    def test_allmatch(self):

        T = 10
        S = 5
        F = 3

        dynamic = np.random.randn(T, S, F)
        static = np.random.randn(S, F)
        temporal = np.random.uniform(size=(T, F))
        adj = np.random.poisson(size=(S, S))
        df = Dataset(dynamic, static, temporal, adj)
  
    def test_T_dimension_mismatch(self):
        """
        Test that the initializer raises an AssertionError
        when the T dimension (first axis) of dynamic_feats_TSF
        and temp_feats_TF do not match.
        """
        dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS = valid_arrays()
        # Introduce a T-dimension mismatch: Increase T of temp_feats_TF by 1.
        temp_feats_TF = np.zeros((dynamic_feats_TSF.shape[0] + 1, temp_feats_TF.shape[1]))
        msg = "Mismatch in T dimension: dynamic_feats_TSF.shape[0]={} vs temp_feats_TF.shape[0]={}".format(dynamic_feats_TSF.shape[0], temp_feats_TF.shape[0])
        with pytest.raises(AssertionError, match=re.escape(msg)):
            Dataset(dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS)

    def test_S_dimension_mismatch_dynamic_vs_static(self):
        """
        Test that an AssertionError is raised when the S dimension
        of dynamic_feats_TSF (axis 1) and static_feats_SF (axis 0) do not match.
        """
        dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS = valid_arrays()
        # Introduce an S-dimension mismatch: Change static_feats_SF so S is off by one.
        static_feats_SF = np.zeros((static_feats_SF.shape[0] + 1, static_feats_SF.shape[1]))
        with pytest.raises(AssertionError):
            Dataset(dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS)

    def test_S_dimension_mismatch_adj(self):
        """
        Test that an AssertionError is raised when the adjacency matrix
        does not have matching S dimensions (i.e. is not square or matching S).
        """
        dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS = valid_arrays()
        # Introduce an S-dimension mismatch in the adjacency matrix: Make it non-square.
        adj_SS = np.zeros((adj_SS.shape[0], adj_SS.shape[1] + 1))
        with pytest.raises(AssertionError):
            Dataset(dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS)

    def test_F_dimension_mismatch_static(self):
        """
        Test that an AssertionError is raised when the F dimension
        of dynamic_feats_TSF (axis 2) does not match static_feats_SF (axis 1).
        """
        dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS = valid_arrays()
        # Introduce an F-dimension mismatch: Change static_feats_SF’s F dimension.
        static_feats_SF = np.zeros((static_feats_SF.shape[0], static_feats_SF.shape[1] + 1))
        with pytest.raises(AssertionError):
            Dataset(dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS)

    def test_F_dimension_mismatch_temp(self):
        """
        Test that an AssertionError is raised when the F dimension
        of dynamic_feats_TSF (axis 2) does not match temp_feats_TF (axis 1).
        """
        dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS = valid_arrays()
        # Introduce an F-dimension mismatch: Change temp_feats_TF’s F dimension.
        temp_feats_TF = np.zeros((temp_feats_TF.shape[0], temp_feats_TF.shape[1] + 1))
        with pytest.raises(AssertionError):
            Dataset(dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS)


if __name__ == '__main__':
    unittest.main()
