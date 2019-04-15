
import numpy as np

from .cdist.gamma import GammaABTest
from .cdist.gamma import GammaModel


class PoissonModel(GammaModel):
    """
    """
    def __init__(self, shape=0.001, rate=0.001):
        super().__init__(shape, rate)

    def update(self, data):
        """
        Update posterior parameters with new data.
        
        Parameters
        ----------
        data : array-like, shape = (n_samples)
        """
        self._shape_posterior += np.sum(data)
        self._rate_posterior += len(data)


class PoissonABTest(GammaABTest):
    """
    """
    def __init__(self, modelA, modelB, simulations=1000000, random_state=None):
        super().__init__(modelA, modelB, simulations, random_state)

        if not all(isinstance(m, PoissonModel) for m in [modelA, modelB]):
            raise TypeError("")
