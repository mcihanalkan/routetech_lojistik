from typing import Optional, List
from sklearn.pipeline import Pipeline

from src.missing import DataPreprocessor
from src.scaling import FeatureScaler
from src.encoding import CategoricalEncoder
from src.utils import get_column_transformer_pipeline


def get_preprocessing_pipeline(
    missing_strategy: str = "mean",
    missing_columns: Optional[List[str]] = None,
    encode_method: str = "onehot",
    encode_columns: Optional[List[str]] = None,
    encode_drop: Optional[str] = None,
    scale_method: str = "minmax",
    scale_columns: Optional[List[str]] = None,
    logging_enabled: bool = False
) -> Pipeline:
    """
    ⭐ DOĞRU SIRADA pipeline:
    1. Missing values
    2. Encoding (categorical → numeric)
    3. Scaling (normalization)
    """
    
    steps = []
    
    # Step 1: Missing values
    steps.append((
        'missing_values',
        DataPreprocessor(
            strategy=missing_strategy,
            columns=missing_columns,
            logging_enabled=logging_enabled
        )
    ))
    
    # Step 2: Encoding 
    if encode_columns:
        steps.append((
            'encoding',
            CategoricalEncoder(
                method=encode_method,
                columns=encode_columns,
                logging_enabled=logging_enabled,
                drop=encode_drop
            )
        ))
    
    # Step 3: Scaling 
    if scale_columns:
        steps.append((
            'scaling',
            FeatureScaler(
                method=scale_method,
                columns=scale_columns,
                logging_enabled=logging_enabled
            )
        ))
    
    return Pipeline(steps)


__all__ = [
    'get_preprocessing_pipeline',
    'get_column_transformer_pipeline',
]