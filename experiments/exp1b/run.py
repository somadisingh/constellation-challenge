"""Execute the predeclared matrix, freeze inner choices, then outer evaluation."""
from .train import *
from experiments.exp1 import SCENES

def run():
    pin_threads();assert torch.backends.mps.is_available()
    write_json(OUT/'config.json',CONFIG);write_json(OUT/'training_source_hashes.json',code_fingerprint())
    for fold in SCENES:
        for arm in 'ABCDE':run_arm(fold,arm)
    selection={}
    for fold in SCENES:
        results={a:read_json(OUT/'runs'/fold/f'{a}_s31004'/'result.json') for a in 'ABCDE'}
        arm=max('ABCDE',key=lambda a:(results[a]['best']['metric'],-ord(a)))
        r=results[arm];gate=r['best']['metric']>r['inner_C1']['equal_sky_top1_localization_reward']+1e-9
        selection[fold]={'arm':arm,'step':r['best']['step'],'metric':r['best']['metric'],'inner_C1':r['inner_C1']['equal_sky_top1_localization_reward'],
                         'second_seed_required':gate,'basis':CONFIG['selection']}
    write_json(OUT/'selection_frozen.json',selection)
    for fold,s in selection.items():
        if s['second_seed_required']:run_arm(fold,s['arm'],31005)
    # Delayed import permits report/evaluation utilities to be implemented while
    # matrix runs without changing the immutable training code.
    from .evaluate import run_evaluation
    run_evaluation()
if __name__=='__main__':run()
