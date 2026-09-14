"""Exact legacy training reproduction, output redirected to Exp1B."""
from pathlib import Path
import numpy as np
from experiments.exp1.env import pin_threads, write_json
pin_threads()
from experiments.exp1.config import load_config
from experiments.exp1.stages import Paths
from experiments.exp1.inner import load_inner_aligned
from experiments.exp1.training import train_arm, TripletSampler
from experiments.exp1.env import derive_seed

def run():
    old=Paths(Path('outputs/exp1')); new=Paths(Path('outputs/exp1b/legacy'))
    cfg=load_config(); fold='pisces'; allowed=('scorpius','taurus')
    tensors={s:dict(np.load(f'outputs/exp1/tensors/{s}.npz',allow_pickle=True)) for s in allowed}
    # Replay the deterministic sampler to count sources that actually enter updates.
    sampler=TripletSampler(tensors,cfg,derive_seed(31004,fold,'hardnet','screen_s31004','sampler'),hard=True)
    used={s:set() for s in allowed}
    # Refresh uses no RNG; source choices are unchanged by network ranking.
    for step in range(2000):
        batch=sampler.sample(64,step<500)
        for s,ids in batch['indices'].items(): used[s].update(ids)
    write_json(Path(new.root)/'actual_coverage.json',{s:{'unique_centres':len(v),'unique_query_images':len({tensors[s]['queries'][i].tobytes() for i in v})} for s,v in used.items()})
    train_arm(new,cfg,fold,'hardnet',device='mps',seed=31004,total_steps=2000,
              tag='screen_s31004',tensors=tensors,inner_aligned=load_inner_aligned(old,fold))
if __name__=='__main__':run()
