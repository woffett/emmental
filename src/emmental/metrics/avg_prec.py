"""Emmental avg prec scorer."""
import logging
from typing import Dict, List, Optional

from numpy import ndarray
from sklearn.metrics import average_precision_score

from emmental.utils.utils import pred_to_prob, prob_to_pred

logger = logging.getLogger(__name__)


def avg_prec_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Average Precision.

    Args:
      golds: Ground truth values.
      probs: Predicted probabilities.
      preds: Predicted values.
      uids: Unique ids, defaults to None.
      pos_label: The positive class label, defaults to 1.

    Returns:
      AP score.
    """
    if len(probs.shape) == 2 and probs.shape[1] == 1:
        probs = probs.reshape(probs.shape[0])

    if len(golds.shape) == 2 and golds.shape[1] == 1:
        golds = golds.reshape(golds.shape[0])

    if len(probs.shape) > 1:
        if len(golds.shape) > 1:
            golds = pred_to_prob(prob_to_pred(golds), n_classes=probs.shape[1])
        else:
            golds = pred_to_prob(golds, n_classes=probs.shape[1])
    else:
        if len(golds.shape) > 1:
            golds = prob_to_pred(golds)

    try:
        avg_prec = average_precision_score(golds, probs)
    except ValueError:
        logger.warning(
            "Only one class present in golds."
            "AP score is not defined in that case, set as nan instead."
        )
        avg_prec = float("nan")

    return {"avg_prec": avg_prec}
