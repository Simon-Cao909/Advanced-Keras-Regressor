from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score

from .estimator import SKGraphEstimator

class SKGraphRegressor(SKGraphEstimator, RegressorMixin):
    '''
    SKGraphRegressor is the regressor branch of SKGraphEstimator
    The only thing different is that it now supports .score()
    '''
    scoring_func = r2_score
    must_be_vector = True