"""How much resolving power does the three-scene total score actually have?

Identification contributes 0.30 weight and takes values in {0, 1/3, 2/3, 1} over three
scenes, so a single scene flipping moves the equal-scene mean by 0.10. Localization is
weighted 0.20 over 71 queries and recovery 0.25 over 26 figure points, so realistic
changes there are an order of magnitude smaller. This quantifies the mismatch, which
explains why three appearance experiments that improved their own target metric all
lost on the total.

Consequence for method selection: appearance changes must be judged on candidate
recall and per-category top-1, with identification held out to the synthetic
benchmark. Selecting appearance changes on the three-scene total is not viable.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constellation.contracts import read_truth
from lab.cache import TRAIN, ROOT

W = {'presence': .25, 'localization': .20, 'recovery': .25, 'identification': .30}


def main():
    truth = read_truth(ROOT / 'train_ground_truth.csv')
    n_present = {n: sum(p is not None for p in truth[n].patches) for n in TRAIN}
    n_figure = {n: sum(1 for p in truth[n].patches if p and p[2] == 1) for n in TRAIN}
    print(f'truth-present per scene: {n_present}  (total {sum(n_present.values())})')
    print(f'figure points per scene: {n_figure}  (total {sum(n_figure.values())})\n')

    print('effect on the equal-scene weighted mean of one minimal change:')
    idq = W['identification'] * (1 / 3)
    print(f'  one scene changes identification      {idq:.4f}')
    for scene in TRAIN:
        loc1 = W['localization'] * (1 / 3) * (1. / n_present[scene])
        rec1 = W['recovery'] * (1 / 3) * (1. / n_figure[scene])
        print(f'  one query fully fixed in {scene:9s}    localization {loc1:.4f}   '
              f'recovery {rec1:.4f}')

    mean_loc1 = np.mean([W['localization'] * (1/3) / n_present[n] for n in TRAIN])
    mean_rec1 = np.mean([W['recovery'] * (1/3) / n_figure[n] for n in TRAIN])
    print(f'\n  identification quantum / one-query localization = '
          f'{idq/mean_loc1:.1f}x')
    print(f'  identification quantum / one-query recovery     = '
          f'{idq/mean_rec1:.1f}x')

    print('\nthe seven recoverable candidates are worth, if all became exact:')
    gain = sum(W['localization'] * (1/3) * (1. / n_present[n]) *
               c for n, c in (('pisces', 2), ('scorpius', 3), ('taurus', 2)))
    print(f'  localization contribution {gain:.4f} '
          f'= {gain/idq:.2f} identification quanta')
    print('\nSo a real 7-query localization repair is worth about one third of the '
          'noise introduced by a single scene changing its class name.')


if __name__ == '__main__':
    main()
