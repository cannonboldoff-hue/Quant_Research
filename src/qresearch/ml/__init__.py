"""ML trade-classification models. See notebooks/14_machine_learning for the
original research; ``models.py`` consolidates the reusable pieces."""
from .models import (
    train_classifier, predict_signals, label_trade_outcome, label_signals,
    align_signals_to_ohlcv, get_score, lgbm_objective, walk_forward_topn,
)
