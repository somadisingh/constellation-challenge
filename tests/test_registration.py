import unittest
import cv2
import numpy as np
from constellation.retrieval import verify
from constellation.refine import refine_candidates

class RegistrationTests(unittest.TestCase):
    def test_rotated_scaled_center(self):
        cv2.setNumThreads(1)
        rng=np.random.default_rng(6643)
        image=cv2.GaussianBlur(rng.normal(100,25,(128,128)).astype(np.float32),(0,0),.7)
        yy,xx=np.mgrid[:32,:32].astype(np.float32)-15.5
        angle=np.deg2rad(30);scale=1.15;x,y=65.5,59.5
        mx=(x+scale*(np.cos(angle)*xx+np.sin(angle)*yy)).astype(np.float32)
        my=(y+scale*(-np.sin(angle)*xx+np.cos(angle)*yy)).astype(np.float32)
        q=cv2.remap(image,mx,my,cv2.INTER_LINEAR)
        q=(q*.8+12).astype(np.float32)
        candidates=np.array([[x,y],[30,30],[90,90]],np.float32)
        matches=verify(image,q,candidates,keep=3)
        self.assertLess(np.linalg.norm(np.array(matches[0][:2])-[x,y]),2)
        fine=refine_candidates(image,q,matches)
        self.assertLess(np.linalg.norm(np.array(fine[0][:2])-[x,y]),2)
    def test_quad_affine_invariance(self):
        from constellation.quad import quads
        p=np.array([[0,0],[100,20],[70,170],[10,80]])
        i,d=quads(p);j,e=quads(p@np.array([[-2.,.3],[.8,3.]])+[700,900])
        np.testing.assert_array_equal(i,j);np.testing.assert_allclose(d,e,atol=1e-8)
