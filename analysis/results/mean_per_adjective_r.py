"""
Mean per-adjective Pearson r of a predictions_vs_truth.csv (as written by
model/src/train.py or experiments/sparsh_x/train_sparshimg.py).

Per adjective: Pearson r between the true and the predicted expected rating
over all objects in the CSV (85 for model/src runs; the Sparsh-X run's CSV has
75, since train_sparshimg.py excludes objects without Sparsh-X image features);
then the mean over the 18 adjectives. The Pearson
function and the nan-mean are copied from the original
Training_Phase2/analyze_predictions.py (pearson(); mean_r = np.nanmean(...)).

Prints only; writes no files.

Usage: python mean_per_adjective_r.py --csv path/to/predictions_vs_truth.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--csv', required=True, help='predictions_vs_truth.csv to analyze')
    args = p.parse_args()

    per_adj = defaultdict(lambda: {'true': [], 'pred': []})
    adj_order = []
    with open(args.csv, newline='') as f:
        for row in csv.DictReader(f):
            adj = row['adjective']
            if adj not in per_adj:
                adj_order.append(adj)
            per_adj[adj]['true'].append(float(row['true_expected_rating']))
            per_adj[adj]['pred'].append(float(row['predicted_expected_rating']))
    if not adj_order:
        raise SystemExit(f'No rows found in {args.csv}')

    print(f'{"adjective":15s} {"n_objects":>9s} {"pearson_r":>10s}')
    rs = []
    for adj in adj_order:
        t = np.array(per_adj[adj]['true'])
        p_ = np.array(per_adj[adj]['pred'])
        r = pearson(t, p_)
        rs.append(r)
        print(f'{adj:15s} {len(t):9d} {r:10.4f}')

    mean_r = float(np.nanmean(rs))
    print(f'\nmean Pearson r over {len(rs)} adjectives = {mean_r:.4f}')


if __name__ == '__main__':
    main()
