"""Benchmark integrity tests use known geometry; they do not learn from Kaggle labels."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from lab.imagebench.generate import (sample_maps, degrade, source_regions, region_image,
    rng_for, add_sources, place_template, PROFILES, draw_params, BuildConfig,
    real_scene, save_scene, sha)
from lab.imagebench.audit import overlap
from lab.imagebench.evaluate import run, measurement_rows
from constellation.contracts import ScenePrediction
from constellation.pipeline import Config


class ImageBenchmark(unittest.TestCase):
    def test_center_mapping_under_rotation_and_scale(self):
        for a in [0,37,180,273]:
            for s in [.65,1.,1.5]:
                x,y=sample_maps((101.25,99.75),a,s)
                # Average of symmetric central pixels equals the continuous centre.
                self.assertAlmostEqual(float(x[15:17,15:17].mean()),101.25,places=4)
                self.assertAlmostEqual(float(y[15:17,15:17].mean()),99.75,places=4)

    def test_identity_crop_and_recorded_transform(self):
        image=np.arange(128*128,dtype=np.uint8).reshape(128,128)
        p=dict(angle=0.,scale=1.,blur=0.,gain=1.,offset=0.,drift=0.,drift_angle=0.,noise=0.,shot=0.,jpeg=100)
        q,meta=degrade(image,(63.5,63.5),p,np.random.default_rng(1))
        np.testing.assert_array_equal(q,image[48:80,48:80])
        np.testing.assert_allclose(np.array(meta['patch_to_source'])@[15.5,15.5,1],[63.5,63.5])

    def test_pixel_ramp_verifies_rotation_direction(self):
        yy,xx=np.mgrid[:160,:160]; image=(xx*.5+yy*.25).astype(np.float32)
        p=dict(angle=90.,scale=1.,blur=0.,gain=1.,offset=0.,drift=0.,drift_angle=0.,noise=0.,shot=0.,jpeg=100)
        q,_=degrade(image,(80,80),p,np.random.default_rng(2))
        mx,my=sample_maps((80,80),90,1)
        np.testing.assert_allclose(q,np.rint(.5*mx+.25*my),atol=1)

    def test_invalid_footprint_is_rejected_not_padded(self):
        p=draw_params(np.random.default_rng(1),'stress')
        with self.assertRaises(ValueError):degrade(np.zeros((80,80)),(3,3),p,np.random.default_rng(1))

    def test_region_splits_are_disjoint_with_buffer(self):
        images={f's{i}':np.zeros((3000,3000),np.uint8) for i in range(3)}
        rs=source_regions(images)
        self.assertEqual(len(rs),27)
        for a in rs:
            self.assertEqual(region_image(images,a).shape,(808,808))
            for b in rs:
                if a['source']==b['source'] and a['split']!=b['split']:
                    self.assertFalse(overlap(a['box'],b['box']))
        self.assertEqual({s:sum(r['split']==s for r in rs) for s in ['development','calibration','confirmation']},
                         dict(development=9,calibration=9,confirmation=9))

    def test_seeds_independent_of_execution_order_and_split(self):
        a=rng_for(12,'real','development',4).normal(size=10)
        rng_for(12,'rendered','development',5).normal(size=100)
        np.testing.assert_array_equal(a,rng_for(12,'real','development',4).normal(size=10))
        self.assertFalse(np.array_equal(a,rng_for(12,'real','confirmation',4).normal(size=10)))

    def test_psf_renderer_repeatable(self):
        a=np.zeros((160,160),np.float32);b=a.copy()
        ra=add_sources(a,[[80.2,79.8],[85.1,80]],np.random.default_rng(2))
        rb=add_sources(b,[[80.2,79.8],[85.1,80]],np.random.default_rng(2))
        np.testing.assert_array_equal(a,b);self.assertEqual(ra,rb)
        self.assertGreater(a.max(),0)

    def test_reflection_affine_placement_and_model_error_separate(self):
        p=np.array([[0,0],[1,2],[3,1],[4,3]],float)
        xy,meta=place_template(p,3000,np.random.default_rng(8),0.)
        np.testing.assert_allclose(xy,meta['ideal_nodes'])
        norm=(p-p.mean(0))/np.maximum(np.ptp(p,axis=0),1)
        np.testing.assert_allclose(xy,norm@meta['affine']+meta['translation'])

    def test_confirmation_requires_explicit_access(self):
        with self.assertRaisesRegex(ValueError,'Confirmation'):
            run(Path('/nonexistent'),Path('/nonexistent'),'confirmation','real',1,Config())

    def test_metrics_distinguish_branch_union_and_presence(self):
        e=dict(id='x',track='rendered',split='development')
        lab=dict(patches=[[100,100,1],None],queries=[dict(category='figure',profile='mild',physical_id='a'),
                                                            dict(category='absent',profile='mild',physical_id='b')])
        pred=ScenePrediction([None,None])
        rows=measurement_rows(e,lab,pred,[[[100,100]],[[2,2]]],[[[100,100,.8]],[[2,2,.2]]],
                              [[[200,200,.7]],[[2,2,.2]]])
        self.assertEqual(rows[0]['localization'],0.)
        self.assertTrue(rows[0]['union_recall12'])
        self.assertFalse(rows[0]['refined_recall12'])
        self.assertNotIn('localization',rows[1])

if __name__=='__main__':unittest.main()

class ImageBenchmarkDataset(unittest.TestCase):
    def fixture(self):
        from lab.imagebench.generate import rendered_scene
        images={'source':np.random.default_rng(4).integers(0,50,(700,700),dtype=np.uint8)}
        region=dict(id='r',source='source',box=[0,0,700,700],split='development',guard=0)
        patterns={'a':np.array([[0.,0.],[1,0],[0,1],[1,1],[.5,2]]),
                  'b':np.array([[0.,0.],[1,0],[.5,1]])}
        return rendered_scene(images,region,patterns,'a',512,np.random.default_rng(44),'stress')

    def test_rendered_labels_include_partial_clutter_absent_and_duplicates(self):
        image,patches,labels=self.fixture()
        self.assertEqual(image.shape,(512,512))
        self.assertTrue(all(p.shape==(32,32) and p.dtype==np.uint8 for p in patches))
        self.assertLess(labels['issued_unique'],labels['reference_nodes'])
        self.assertEqual({q['category'] for q in labels['queries']},{'figure','off-figure','absent'})
        groups={}
        for q,t in zip(labels['queries'],labels['patches']):
            if t is not None:
                if q['physical_id'] in groups: self.assertEqual(groups[q['physical_id']],t)
                groups[q['physical_id']]=t
                np.testing.assert_allclose(np.array(q['patch_to_source'])@[15.5,15.5,1],t[:2])
        self.assertLess(len(groups),sum(t is not None for t in labels['patches']))

    def test_rendered_reproduction_and_artifact_hashes(self):
        a,aq,al=self.fixture();b,bq,bl=self.fixture()
        np.testing.assert_array_equal(a,b);self.assertEqual(al,bl)
        for x,y in zip(aq,bq):np.testing.assert_array_equal(x,y)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            entry=save_scene(root,dict(id='x',split='development',track='rendered'),a,aq,al)
            self.assertTrue(all(sha(root/f['path'])==f['sha256'] for f in entry['files']))
            self.assertEqual(sha(root/entry['labels']),entry['labels_sha256'])

    def test_style_selection_preserves_pixels_and_is_reproducible(self):
        from lab.imagebench.realism import choose_sources, FEATURES
        from lab.imagebench.generate import patch_stats
        import cv2
        image=np.zeros((100,160),np.uint8)
        image[:,80:]=120
        points=np.array([[40.,50.],[120.,50.]])
        stats=patch_stats(cv2.getRectSubPix(image,(32,32),(120.,50.)))
        model={'rows':[{'stats':stats}]}
        before=image.copy()
        a=choose_sources(image,points,20,np.random.default_rng(4),model)
        b=choose_sources(image,points,20,np.random.default_rng(4),model)
        self.assertEqual(a,b)
        self.assertEqual(a,[1]*20)
        np.testing.assert_array_equal(before,image)

    def test_style_selection_rejects_empty_candidate_set(self):
        from lab.imagebench.realism import choose_sources
        with self.assertRaises(ValueError):
            choose_sources(np.zeros((32,32),np.uint8),[],1,np.random.default_rng(1),{'rows':[]})
