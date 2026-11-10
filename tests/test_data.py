import pandas as pd

from credit_default.data import clean
from credit_default.metrics import best_threshold, ks_statistic


def test_clean_rules():
    df = pd.DataFrame(
        {
            "person_age": [25, 144, 30, 25],
            "person_emp_length": [3, 2, 40, 3],
            "loan_status": [0, 1, 1, 0],
        }
    )
    out, rep = clean(df)
    assert rep["duplicates_removed"] == 1
    assert rep["age_over_100_removed"] == 1
    assert rep["emp_length_set_to_nan"] == 1  # 40 ans d'ancienneté à 30 ans
    assert len(out) == 2


def test_ks_perfect_separation():
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_threshold_favours_recall_when_fn_costly():
    y = [0] * 80 + [1] * 20
    p = [i / 100 for i in range(100)]
    t_cheap, _ = best_threshold(y, p, cost_fn=1, cost_fp=1)
    t_costly, _ = best_threshold(y, p, cost_fn=10, cost_fp=1)
    assert t_costly <= t_cheap
