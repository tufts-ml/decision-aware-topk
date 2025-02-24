import numpy as np
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from scipy.stats import skewnorm, rv_continuous

# Define a scipy stats mixture of skew normals
class MixtureModel(scipy.stats.rv_continuous):
    def __init__(self, submodels, weights = None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf
            
    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x)  * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x)  * weight
        return cdf

    def _rvs(self, size, random_state=None):
        submodel_choices = random_state.choice(len(self.submodels), size=size, p=self.weights)
        submodel_samples = [submodel.rvs(size=size, random_state=random_state)
                             for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


class QuantizedDist(scipy.stats.rv_continuous):
    
    def __init__(self, dist,*args,  **kwargs):
        self.dist = dist
        super().__init__(*args, **kwargs)
    
    def rvs(self, *args, **kwargs):
        vals = np.atleast_1d(np.round(self.dist.rvs(*args, **kwargs)))
        return np.maximum(0, vals)
    
    
class QuantizedNormal(QuantizedDist):
    
    def __init__(self, loc=0, scale=1):
        super().__init__(scipy.stats.norm(loc=loc, scale=scale))
