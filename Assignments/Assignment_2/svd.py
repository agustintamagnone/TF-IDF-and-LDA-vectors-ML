from collections import defaultdict

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.accuracy import mae
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')

TEST_SIZE = 0.25
train_set, test_set = train_test_split(data, test_size=TEST_SIZE, random_state=20)

def precision_recall_at_n(predictions, n=10, threshold=3.5):
    """Return precision and recall at n metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of relevant and recommended items in top n
        n_rel_and_rec = sum(
            (true_r >= threshold)
            for (_, true_r) in user_ratings[:n]
        )

        # Precision@n: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec / n

        # Recall@n: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec / n_rel if n_rel != 0 else 0

    return precisions, recalls

def f1(prec, rec): return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def eval_svd(prec_recall = False,
             n = 10,
             verbose = False):
    algo = SVD(random_state=20)
    algo.fit(train_set)

    predictions = algo.test(test_set)

    if prec_recall:
        return precision_recall_at_n(predictions, n, threshold=4.0)
    return mae(predictions, verbose)

def eval_all_ns_svd(verbose = True):
    for n in range(10, 101, 10):
        prec, rec = eval_svd(prec_recall=True, n=n)

        avg_prec = sum(p for p in prec.values()) / len(prec)
        avg_rec = sum(r for r in rec.values()) / len(rec)
        f1_ = f1(avg_prec, avg_rec)

        if verbose:
            print(f"For N={n} (SVD): (\n"
                  f"\tPrecision: {avg_prec}\n"
                  f"\tRecall: {avg_rec}\n"
                  f"\tF1: {f1_}\n)\n")

if __name__ == "__main__":
    # Activity 2: MAE comparison (KNN result you take from knn.py output)
    mae_svd = eval_svd(prec_recall=False, verbose=True)
    print(f"MAE SVD: {mae_svd}")

    # Activity 3: Top-N for SVD
    eval_all_ns_svd(verbose=True)

"""
OUTPUT TESTING ALL Ns SVD TEST_SIZE = 0.25

MAE SVD: 0.7376398383696386

For N=10 (SVD): (
    Precision: 0.6285259809119816
    Recall: 0.6611825990803654
    F1: 0.6444408420565615
)

For N=20 (SVD): (
    Precision: 0.48075291622481403
    Recall: 0.8308572819312126
    F1: 0.6090789196616732
)

For N=30 (SVD): (
    Precision: 0.38624955814775697
    Recall: 0.9025043539794722
    F1: 0.5409751305827021
)

For N=40 (SVD): (
    Precision: 0.3214740190880165
    Recall: 0.9402145227661277
    F1: 0.4791270292339553
)

For N=50 (SVD): (
    Precision: 0.27363732767762483
    Recall: 0.9622498787366661
    F1: 0.4261027770318299
)

For N=60 (SVD): (
    Precision: 0.2362318840579721
    Recall: 0.9739833046832429
    F1: 0.38023966852647484
)

For N=70 (SVD): (
    Precision: 0.20672625359794025
    Recall: 0.980189613000145
    F1: 0.34144109488048485
)

For N=80 (SVD): (
    Precision: 0.18295334040296932
    Recall: 0.9833031750249449
    F1: 0.3085060586926748
)

For N=90 (SVD): (
    Precision: 0.16356780959113965
    Recall: 0.9848566722149084
    F1: 0.28054234507797166
)

For N=100 (SVD): (
    Precision: 0.14775185577942745
    Recall: 0.9857464885467542
    F1: 0.25698471239923615
)
"""

"""
OUTPUT TESTING ALL Ns SVD TEST_SIZE = 0.75

MAE SVD: 0.7727457290866755

For N=10 (SVD): (
    Precision: 0.7510074231177075
    Recall: 0.32910470496506683
    F1: 0.45765633027460484
)

For N=20 (SVD): (
    Precision: 0.6846235418875923
    Recall: 0.547461075867437
    F1: 0.6084074671573858
)

For N=30 (SVD): (
    Precision: 0.6154471544715442
    Recall: 0.6660888426696713
    F1: 0.639767409984225
)

For N=40 (SVD): (
    Precision: 0.5577942735949096
    Recall: 0.7430306399421951
    F1: 0.6372237058995959
)

For N=50 (SVD): (
    Precision: 0.5087804878048784
    Recall: 0.7949438969991977
    F1: 0.6204562074729375
)

For N=60 (SVD): (
    Precision: 0.4684340756451045
    Recall: 0.834231356118647
    F1: 0.5999735383297894
)

For N=70 (SVD): (
    Precision: 0.4338130586274811
    Recall: 0.864475295167951
    F1: 0.5777155295406654
)

For N=80 (SVD): (
    Precision: 0.40469247083775206
    Recall: 0.8900033200993815
    F1: 0.556389609336902
)

For N=90 (SVD): (
    Precision: 0.3789678331565928
    Recall: 0.9104110082420427
    F1: 0.5351669749771046
)

For N=100 (SVD): (
    Precision: 0.35536585365853635
    Recall: 0.9258334213512842
    F1: 0.5135962695913647
)
"""