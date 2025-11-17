from collections import defaultdict

import pandas as pd
import numpy as np
from surprise import KNNWithMeans, Reader
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')

TEST_SIZE = 0.25
train_set, test_set = train_test_split(data, test_size=TEST_SIZE, random_state=20)

similarity_opts = {'name': "pearson", 'user_based': True}

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

def f1(prec, rec): return 2 * (prec * rec) / (prec + rec)

def eval_k(test_k: int = 10, 
           prec_recall=False, # Return precision, recall instead of mae
           n=10, # For precision recall
           verbose=False):
    algo = KNNWithMeans(test_k, sim_options=similarity_opts, verbose=False)
    algo.fit(train_set)

    predictions = algo.test(test_set)

    if prec_recall:
        return precision_recall_at_n(predictions, n, threshold=4.0)
    return mae(predictions, verbose)


def eval_all_ks(verbose=False):
    results = 10000000, 0 # mae, k
    # best found is (0.7407265872957481, 68) at 0.25
    # best found is (0.8129837688720589, 72) at 0.75
    if verbose: print("MAE For Ks")
    for k in range(10, 101):
        err = eval_k(k, verbose=False)

        if verbose:
            print(err)
        if err < results[0]:
            results = err, k

    if verbose:
        print(results)
    return results

def eval_all_ns(k=68, verbose=True):
    for n in range(10, 101, 10):
        prec, rec = eval_k(k, True, n)

        avg_prec = sum(p for p in prec.values()) / len(prec)
        avg_rec = sum(r for r in rec.values()) / len(rec)
        f1_ = f1(avg_prec, avg_rec)

        if verbose:
            print(f"For N={n}: (\n\tPrecision: {avg_prec}\n\tRecall: {avg_rec}\n\tF1: {f1_}\n)\n")

"""
OUTPUT TESTING ALL Ns

For N=10: (
        Precision: 0.629374337221633
        Recall: 0.6595663617663685
        F1: 0.6441167419359395
)

For N=20: (
        Precision: 0.48218451749734886
        Recall: 0.8314004869422419
        F1: 0.6103730497659497
)

For N=30: (
        Precision: 0.388016967126193
        Recall: 0.9047672238252851
        F1: 0.5431146769910478
)

For N=40: (
        Precision: 0.3225609756097561
        Recall: 0.942372262534231
        F1: 0.4806143237039577
)

For N=50: (
        Precision: 0.2746977730646872
        Recall: 0.9637372687400005
        F1: 0.4275338998104245
)

For N=60: (
        Precision: 0.23695652173913043
        Recall: 0.9747633740191073
        F1: 0.3812375111357437
)

For N=70: (
        Precision: 0.20719587941221027
        Recall: 0.9808251738578005
        F1: 0.34212008934981886
)

For N=80: (
        Precision: 0.18331124072110289
        Recall: 0.9840402712400074
        F1: 0.3090511147537638
)

For N=90: (
        Precision: 0.163909508660304
        Recall: 0.9856022749708473
        F1: 0.281075125849752
)

For N=100: (
        Precision: 0.14798515376458113
        Recall: 0.9864246052310282
        F1: 0.2573606154649538
)
"""

# eval_all_ks(verbose=True)
eval_all_ns(k=68, verbose=True)


