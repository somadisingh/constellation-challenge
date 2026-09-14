"""Regression tests for the causal controls in Experiment 1B."""
import copy
import numpy as np
import unittest
from unittest.mock import patch
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    # torch is not installed in the production .venv (Python 3.14); the two
    # tests below that need it are individually skipped rather than failing
    # the whole module's collection at import time.
    torch = None
    HAVE_TORCH = False
from experiments.exp1b.stream import choose_indices,eligible,IGNORE,AnchorStream,NegativePool
from experiments.exp1b.data import generate,local_context,valid_source,source_weights,stats,OUT,OLD
if HAVE_TORCH:
    from experiments.exp1b.train import select_negative_distance
from experiments.exp1.splits import cell_box,partition_records
from experiments.exp1.pose import sample_query_frame,query_to_source,aligned_candidate
from experiments.exp1.env import read_json,pin_threads
pin_threads()

def test_random_selection_ignores_scores():
    valid=np.ones(30,bool);groups=np.arange(30)//3
    a,_=choose_indices(np.arange(30),np.arange(30),valid,np.random.default_rng(6),True,groups)
    b,_=choose_indices(-np.arange(30),np.sin(np.arange(30)),valid,np.random.default_rng(6),True,groups)
    assert np.array_equal(a,b)
    assert len(set(groups[a]))==3

def test_hard_sources_are_distinct_and_hard():
    nc=np.arange(30);net=-nc;groups=np.arange(30)//3
    ids,origins=choose_indices(nc,net,np.ones(30,bool),np.random.default_rng(1),False,groups)
    assert ids[0]==29 and ids[1]==0
    assert origins==['classical','network','random'] and len(set(groups[ids]))==3

def test_ambiguity_mask():
    assert eligible([[0,0],[36,0],[72,0],[73,0]],[0,0]).tolist()==[False,False,False,True]

@unittest.skipUnless(HAVE_TORCH, 'torch not available in this environment')
def test_random_loss_has_no_hidden_hardest_selection():
    d=torch.tensor([[.1,.7,1.],[.9,.3,.6]])
    ids=torch.tensor([2,0]);assert torch.allclose(select_negative_distance(d,True,ids),torch.tensor([1.,.9]))
    assert torch.allclose(select_negative_distance(d,False,ids),torch.tensor([.1,.3]))

def test_random_control_uniform_over_locations():
    # Different numbers of poses must not bias physical location probability.
    groups=np.repeat(np.arange(4),[1,2,3,4]);counts=np.zeros(4)
    rng=np.random.default_rng(99)
    for _ in range(1000):
        ids,_=choose_indices(np.zeros(10),np.zeros(10),np.ones(10,bool),rng,True,groups)
        counts[groups[ids[0]]]+=1
    assert np.max(np.abs(counts-250))<50

def test_local_warp_matches_full_frame_without_boundary_leakage():
    image=np.random.default_rng(8).uniform(0,255,(300,300)).astype(np.float32)
    xy=[145.25,152.5];local,c=local_context(image,xy)
    a,_=aligned_candidate(image,xy,(37,1.2));b,_=aligned_candidate(local,c,(37,1.2))
    # OpenCV float maps differ under coordinate translation by <0.003 gray levels.
    np.testing.assert_allclose(a,b,atol=.003)

def test_fresh_generation_and_centre_mapping():
    image=np.random.default_rng(4).integers(0,255,(300,300),dtype=np.uint8)
    r={'xy':[150.5,150.25],'source_id':'a:0','scene':'a','source_class':'peak'}
    a=generate(image,r,np.random.default_rng(4));b=generate(image,r,np.random.default_rng(5))
    assert a['hash']!=b['hash'] and a['source_id']==b['source_id']
    np.testing.assert_allclose(query_to_source([[15.5,15.5]],r['xy'],a['true_pose']['angle'],a['true_pose']['scale']),[r['xy']])
    assert np.isfinite(a['p']).all()

def test_fold_manifest_isolation():
    for fold in ('pisces','scorpius','taurus'):
        m=read_json(OUT/'folds'/fold/'manifest.json');r=read_json(OUT/'folds'/fold/'recipe.json')
        assert fold not in m['allowed'] and fold not in r['allowed']
        assert set(m['banks'])==set(m['allowed'])==set(r['diagnostics'])
        for scene,bank in m['banks'].items():
            for part,cells in [('fit',range(6)),('val',[6,7])]:
                recs=partition_records(bank,part)
                assert all(x['cell'] in cells and valid_source(x,(3000,3000)) for x in recs)

def test_brightness_is_not_clipping():
    s=stats(np.full((32,32),220,np.uint8));assert s['bright_gt200']==1 and s['clip255']==0

def test_d_e_have_identical_anchors_despite_different_loss():
    a=AnchorStream('pisces','D',31004);b=AnchorStream('pisces','E',31004)
    for step in range(18):
        x=a.sample(step);y=b.sample(step)
        assert [e['hash'] for e in x]==[e['hash'] for e in y]
        a.record_loss(x,np.ones(64,bool),np.linspace(0,1,64),step)
        b.record_loss(y,np.zeros(64,bool),np.linspace(0,1,64),step)
    assert a.stats()['anchor_digest']==b.stats()['anchor_digest']
    assert len(a.uses)>400 and a.replayed>0
    assert a.stats()['nonzero_loss_presentations']>0 and b.stats()['nonzero_loss_presentations']==0

@unittest.skipUnless(HAVE_TORCH, 'torch not available in this environment')
def test_pool_refresh_introduces_locations():
    import experiments.exp1b.stream as streammod
    # Small real source subset and lightweight deterministic encoder keep this a
    # test of locations/partitions, not GPU training.
    old_count,old_poses=streammod.POOL_CENTRES,streammod.POSES
    streammod.POOL_CENTRES,streammod.POSES=4,[(0,1)]
    class Encoder(torch.nn.Module):
        def forward(self,x):return x.flatten(1)[:,:128]
    a=AnchorStream('pisces','B',31004);p=NegativePool('pisces',31004,a.records,a.images)
    p.refresh(0,Encoder(),'cpu');p.refresh(1,Encoder(),'cpu')
    assert all(r['new_vs_previous']>0 for r in p.logs[2:])
    assert all(r['jaccard_previous']<1 for r in p.logs[2:])
    streammod.POOL_CENTRES,streammod.POSES=old_count,old_poses

def test_cached_a_respects_unit_input_contract():
    a=AnchorStream('pisces','A',31004)
    for rows in a.fixed.values():
        for e in rows:
            assert e['q'].min()>=0 and e['q'].max()<=1
            assert e['p'].min()>=0 and e['p'].max()<=1.000001

def test_complete_aggregation_requires_all_three_skies():
    from experiments.exp1b.evaluate import aggregate_complete,gate_result
    m={k:1. for k in ('presence','localization','recovery','identification','score')}
    folds={'pisces':{'arms':{'A':{'held_out_metrics':m}}}}
    a=aggregate_complete(folds)['A']
    assert not a['complete'] and a['mean'] is None
    assert not gate_result(a,{}, {})['pass']
    for fold in ('scorpius','taurus'):folds[fold]={'arms':{'A':{'held_out_metrics':m}}}
    assert aggregate_complete(folds)['A']['complete']

def test_valid_pose_does_not_mean_correct_pose_or_candidate():
    from experiments.exp1b.evaluate import correct_alignment_state
    a=correct_alignment_state(np.array([[100,100]]),[10,10],np.array([True]))
    assert a['valid_pose'] and not a['near_correct_exists']
    b=correct_alignment_state(np.array([[10,10]]),[10,10],np.array([False]))
    assert b['near_correct_exists'] and not b['valid_pose']

def test_fast_pose_matches_reference():
    from experiments.exp1b.fast_pose import select_pose as fast
    from experiments.exp1.pose import select_pose as reference,SceneReps,prepare_query
    rng=np.random.default_rng(122)
    image=rng.uniform(0,255,(145,145)).astype(np.float32);reps=SceneReps(image)
    for _ in range(20):
        q=rng.uniform(0,1,(32,32)).astype(np.float32);_,qb=prepare_query(q)
        xy=rng.uniform(1,144,2);pose=(rng.uniform(0,360),rng.uniform(.7,1.4))
        a=reference(reps,qb,xy,pose);b=fast(reps,qb,xy,pose)
        assert a['pose']==b['pose'] and a['rejected']==b['rejected']
        if a['ncc'] is not None:assert abs(a['ncc']-b['ncc'])<1e-10

class Experiment1BTests(unittest.TestCase):
    pass
for name,fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
        setattr(Experiment1BTests,name,staticmethod(fn))
del name,fn
if __name__=="__main__":unittest.main()
