"""Packaging invariants, separate from physical parity and the official marking runs."""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from tools import build_combined_submission as pack


class CombinedFileContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.receipt = pack.materialize()

    def test_frozen_sources_and_no_new_controller(self):
        for mode in ('servo', 'torque'):
            name='my_'+mode+'.py'
            self.assertEqual(pack.digest(self.files[name]), pack.RELEASE[name])
        original=(pack.ROOT/'my_detector.py').read_bytes()
        self.assertEqual(self.files['my_detector.py'].count(original), 1)
        self.assertEqual(self.receipt['inputs'][pack.path_for(pack.ENTRY)], pack.R31)
        self.assertFalse(self.receipt['new_detection_rules'])
        self.assertEqual((self.receipt['modules'],self.receipt['normal_tables']),(44,7))

    def test_no_research_imports_runtime_source_or_unbound_data_read(self):
        tree=ast.parse(self.files['my_detector.py'])
        allow={'mujoco','numpy','sim','collections','dataclasses','math','copy','json','types','base64','io','functools'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names:self.assertIn(alias.name.split('.')[0],allow)
            if isinstance(node,ast.ImportFrom):
                self.assertIn(node.module.split('.')[0],allow)
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load):
                self.assertNotIn(node.id,{'open','Path','exec','eval','__file__','__import__'})
        self.assertEqual(set(pack.TABLES),{
            'r21_yaw_impulse/normal_bins','r22_release/model','r22_tangent/model',
            'r24_loaded/normal_response','r24_loaded/dc_drive',
            'r27_load_change/settled_model','r27_load_change/manifold_model'})
        self.assertNotIn('fit_records', str(self.receipt['embedded_runtime_keys']))

    def test_reproducible_archive_no_kit_and_refuse_overwrite(self):
        with TemporaryDirectory() as tmp:
            a,b=(Path(tmp)/n for n in ('a','b'))
            ra=pack.build(a); rb=pack.build(b)
            self.assertEqual(ra,rb)
            za,zb=a/pack.ZIP_NAME,b/pack.ZIP_NAME
            self.assertEqual(za.read_bytes(),zb.read_bytes())
            self.assertEqual(ra['zip_sha256'],hashlib.sha256(za.read_bytes()).hexdigest())
            with ZipFile(za) as z:
                self.assertEqual(sorted(z.namelist()),sorted(pack.RELEASE))
                self.assertIsNone(z.testzip())
                for n in z.namelist():self.assertEqual(z.read(n),self.files[n])
            with self.assertRaises(ValueError):pack.build(a)

    def test_missing_source_in_declared_closure_fails(self):
        with self.assertRaises((KeyError,ValueError)):
            pack.graph({pack.path_for(pack.ENTRY):b'from experiments.missing import Detector'})

class OneUpdatePerSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Engine:
            API='icpbl.shared-estimator.v1';SOURCE='jts'
            def reset(self):
                self.calls=0;self.snapshot=(self.nom,self.nom_box,self.ctrl,self.box_mass,self.box_com)
            def update(self,*args):
                self.calls+=1
                return len(args),self.calls
        class Torque(Engine):
            pass
        ns={'_submission_r10':SimpleNamespace(Detector=Engine),'_submission_torque_class':lambda:Torque}
        exec(compile(pack.DISPATCH,'<packaged-dispatch>','exec'),ns)
        cls.Detector=ns['Detector'];cls.Torque=Torque

    def test_same_instance_can_change_mode_and_does_not_double_update(self):
        obj=self.Detector();obj.nom=object();obj.nom_box=object()
        shared=SimpleNamespace(API=obj.API,SOURCE='jts',reset=lambda:None,update=lambda:None)
        for mode,expected in [('servo','r10'),('torque','r31-torque'),('servo','r10')]:
            obj.ctrl=SimpleNamespace(MODE=mode,shared_estimator=shared);obj.reset()
            self.assertEqual(obj.selected_method,expected)
            self.assertIs(obj.update.__self__,obj._impl)
            self.assertEqual(obj.update(*range(6)),(6,1))
            self.assertEqual(obj.snapshot[2],obj.ctrl)
            self.assertEqual(isinstance(obj._impl,self.Torque),mode=='torque')

    def test_unattached_and_unrecognised_controller_contract(self):
        obj=self.Detector()
        with self.assertRaises(RuntimeError):obj.update(*range(6))
        self.assertFalse(hasattr(obj,'not_present'))
        obj.nom=object();obj.nom_box=object()
        for ctrl in (None,SimpleNamespace(MODE='torque'),SimpleNamespace(MODE='torque',shared_estimator=object())):
            obj.ctrl=ctrl;obj.reset()
            self.assertEqual(obj.selected_method,'r10')
            self.assertFalse(isinstance(obj._impl,self.Torque))

if __name__=='__main__':unittest.main()
