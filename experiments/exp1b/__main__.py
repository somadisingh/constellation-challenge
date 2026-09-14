"""Run Experiment1B locally: prepare, legacy, audit, run, or report."""
import argparse
from experiments.exp1.env import pin_threads

def main():
    pin_threads();p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','legacy','audit','run','integrity','report']);args=p.parse_args()
    if args.stage=='prepare':
        from .data import prepare_fold
        from .validation import fresh_bank
        from experiments.exp1.search import clear_index_cache
        for fold in ('pisces','scorpius','taurus'):
            prepare_fold(fold)
            for step in (0,500,1000,1500,2000):fresh_bank(fold,step)
            clear_index_cache()
    elif args.stage=='legacy':
        from .legacy import run
        run()
    elif args.stage=='audit':
        from .audit import run
        run()
    elif args.stage=='run':
        from .run import run
        run()
    elif args.stage=='integrity':
        from .integrity import run
        run()
    else:
        from .report import make_report
        make_report()
if __name__=='__main__':main()
