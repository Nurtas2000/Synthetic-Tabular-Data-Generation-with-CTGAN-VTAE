from sdv.metrics.tabular import CSTest, KSTest
from sdv.evaluation import evaluate
import pandas as pd

def evaluate_quality(real_data, synthetic_data, metadata):
    """Comprehensive quality evaluation"""
    # Statistical tests
    numerical_cols = real_data.select_dtypes(include='number').columns
    results = {
        'ks_test': {col: KSTest.compute(real_data[col], synthetic_data[col]) 
                   for col in numerical_cols},
        'cs_test': {col: CSTest.compute(real_data[col], synthetic_data[col]) 
                   for col in numerical_cols},
        'sdv_score': evaluate(real_data, synthetic_data, metadata)
    }
    return results
