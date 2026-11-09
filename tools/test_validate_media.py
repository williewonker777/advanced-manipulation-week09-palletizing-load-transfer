import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import validate_media as V

ORIGINAL_READ_OFFICIAL = V.read_official


def fixture(root):
    media=root/'artifacts/media';media.mkdir(parents=True)
    result=[]
    for index,name in enumerate(V.STEMS):
        ctrl='torque' if index==1 else 'servo'
        official=V.read_official(V.ROOT,ctrl)
        summaries={}
        for label,(control,_) in zip(('baseline','ours'),official):
            summaries[label]=dict(tcp_rms_mm=float(control[0]),tcp_max_mm=float(control[1]),
                                  saturation_pct=float(control[2]),false_alarm_rises_in_this_episode=sum(map(int,control[6:9])),
                                  alarm_before_collision=False,detect_delay_ms_raw=None)
        when={2:3.86,3:3.0}.get(index)
        classification=None
        if when is not None:
            classification={}
            for label,(_,collisions) in zip(('baseline','ours'),official):
                val=collisions[(8.,when)][1]
                kind={'경보 중':'alarm_before_collision','놓침':'missed','구분 불가':'indistinguishable'}.get(val,'detected')
                count=6 if index==2 and label=='baseline' else 8 if index==3 and label=='baseline' else 0
                summaries[label]['false_alarm_rises_in_this_episode']=count
                classification[label]=dict(collision_outcome=kind,detect_delay_ms=float(val) if kind=='detected' else None,false_alarms=count)
                summaries[label]['alarm_before_collision']=kind=='alarm_before_collision'
                summaries[label]['detect_delay_ms_raw']=float(val) if kind=='detected' else None
        items=[]
        for suffix,frames,fps in [('mp4',180,15.),('gif',96,8.)]:
            path=media/(name+'.'+suffix);path.write_bytes((name+suffix).encode())
            items.append(dict(file=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),width=1280 if suffix=='mp4' else 640,
                height=720 if suffix=='mp4' else 360,frame_count=frames,fps=fps,duration_s=12.))
        result.append(dict(name=name,condition=dict(box_mass_kg=8.,box_com_m=[.02,-.01,.03],
            collision=None if when is None else dict(time_s=when,force_n=[0.,15.,0.])),
            software_gl=dict(renderer='llvmpipe'),timewarp=[dict(simulation_start_s=0.,simulation_end_s=6.,playback_speed=.5)],
            rendered_frame_count=180,reference_no_collision={label:dict(controller=(ctrl+'_controller') if label=='baseline' else ('my_'+ctrl),
                detector='collision_detector' if label=='baseline' else 'my_detector',box_mass_kg=8.,box_com_m=[.02,-.01,.03],
                false_alarm_rises=sum(map(int,official[j][0][6:9]))) for j,label in enumerate(('baseline','ours'))},
            run_py_equivalent=summaries,
            run_py_validation={label:dict(parsed=dict(summary),matches_direct_run_episode=True) for label,summary in summaries.items()},
            evaluate_classification=classification,artifacts=items))
    data=dict(scenarios=result,total_media_bytes=sum(i['bytes'] for r in result for i in r['artifacts']))
    (media/'media_manifest.json').write_text(json.dumps(data));return data


def fake_probe(path,_):
    mp4=path.suffix=='.mp4'
    return dict(codec='h264' if mp4 else 'gif',width=1280 if mp4 else 640,height=720 if mp4 else 360,
                frame_count=180 if mp4 else 96,fps=15. if mp4 else 8.,duration_s=12.)


def fixture_v2(root):
    media=root/'artifacts/media';media.mkdir(parents=True,exist_ok=True)
    source_names={
        'servo_controller.py','torque_controller.py','collision_detector.py',
        'v1_servo.py','v1_torque.py','v1_detector.py','my_servo.py','my_torque.py',
        'my_detector.py','sim/runner.py','tools/render_v2_comparison.py',
    }
    hashes={}
    for name in source_names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        content=(b'servo-controller' if name in ('v1_servo.py','my_servo.py') else
                 b'torque-controller' if name in ('v1_torque.py','my_torque.py') else
                 name.encode())
        path.write_bytes(content);hashes[name]=hashlib.sha256(content).hexdigest()
    backend=dict(vendor='Mesa/X.org',renderer='llvmpipe (LLVM 20.1)',version='4.5 Mesa')
    records=[]
    for mode,prefix in (('servo','05'),('torque','06')):
        name=f'{prefix}_v2_{mode}_transport_collision'
        variants=V._v2_combinations(mode);summaries={};checks={};references={}
        for index,label in enumerate(V.V2_VARIANTS):
            controller,detector=variants[label];delay=80.-20.*index
            summaries[label]=dict(tcp_rms_mm=2.,tcp_max_mm=4.,saturation_pct=0.,
                torque_jitter=.1,false_alarm_rises_in_this_episode=0,
                false_alarm_times_s=[],detect_delay_ms_raw=round(delay),
                alarm_before_collision=False,diverged=False)
            checks[label]=dict(collision_outcome='detected',collision_label=f'{delay:.0f} ms',
                detect_delay_ms=delay,alarm_before_collision=False,false_alarms=0,
                tool_contact_delay_ms=None)
            references[label]=dict(controller=controller,detector=detector,box_mass_kg=3.7,
                box_com_m=[.02,-.01,.03],false_alarm_rises=0,
                resource_guard_before_reference={'mem_available_gib':10},
                resource_guard_before_collision={'mem_available_gib':10})
        items=[]
        for suffix,frames,fps in [('mp4',180,15.),('gif',96,8.)]:
            path=media/f'{name}.{suffix}';path.write_bytes(f'{name}.{suffix}'.encode())
            items.append(dict(file=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                width=1280 if suffix=='mp4' else 640,height=720 if suffix=='mp4' else 360,
                frame_count=frames,fps=fps,duration_s=12.))
        records.append(dict(name=name,condition=dict(mode=mode,box_mass_kg=3.7,
            box_com_m=[.02,-.01,.03],collision=dict(time_s=3.,force_n=[0.,15.,0.])),
            variants={label:dict(controller=value[0],detector=value[1])
                      for label,value in variants.items()},
            timewarp=[dict(simulation_start_s=0.,simulation_end_s=6.,playback_speed=.5)],
            source_simulation_frames={label:6000 for label in V.V2_VARIANTS},
            rendered_frame_count=180,software_gl=backend,episode_summary=summaries,
            evaluate_classification=checks,reference_no_collision=references,artifacts=items))
    data=dict(schema_version=2,
        generation_command='env -u PYTHONPATH .venv/bin/python tools/render_v2_comparison.py --output-dir artifacts/media --raw-dir "$RAW_DIR" --ffmpeg ffmpeg',
        software_gl=backend,source_hashes=hashes,scenarios=records,
        total_media_bytes=sum(x['bytes'] for row in records for x in row['artifacts']))
    (media/'v2_manifest.json').write_text(json.dumps(data));return data


def fixture_r2(root):
    media=root/'artifacts/media';media.mkdir(parents=True,exist_ok=True)
    source_names={
        'servo_controller.py','torque_controller.py','collision_detector.py',
        'v1_servo.py','v1_torque.py','v1_detector.py',
        'r1_servo.py','r1_torque.py','r1_detector.py',
        'my_servo.py','my_torque.py','my_detector.py','sim/runner.py',
        'tools/render_r2_comparison.py',
    }
    hashes={}
    for name in source_names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        alias=name.replace('my_','r1_') if name.startswith('my_') else name
        content=alias.encode();path.write_bytes(content);hashes[name]=hashlib.sha256(content).hexdigest()
    backend=dict(vendor='Mesa/X.org',renderer='llvmpipe (LLVM 20.1)',version='4.5 Mesa')
    aliases={name:name.replace('my_','r1_') for name in ('my_servo.py','my_torque.py','my_detector.py')}
    records=[]
    for mode,prefix,force in (('servo','07',[0.,15.,0.]),('torque','08',[-15.,0.,0.])):
        name=f'{prefix}_r2_{mode}_comparison';variants=V._r2_combinations(mode)
        summaries={};checks={};references={}
        for index,label in enumerate(V.R2_VARIANTS):
            controller,detector=variants[label];delay=60. if label in ('r1','r2') else 80.-10.*index
            summaries[label]=dict(false_alarm_rises_in_this_episode=0,
                                  detect_delay_ms_raw=round(delay),alarm_before_collision=False)
            checks[label]=dict(collision_outcome='detected',detect_delay_ms=delay,
                               alarm_before_collision=False,false_alarms=0)
            references[label]=dict(controller=controller,detector=detector,box_mass_kg=3.7,
                box_com_m=[.02,-.01,.03],resource_guard_before_reference={'mem_available_gib':10},
                resource_guard_before_collision={'mem_available_gib':10})
        items=[]
        for suffix,frames,fps in [('mp4',180,15.),('gif',96,8.)]:
            path=media/f'{name}.{suffix}';path.write_bytes(f'{name}.{suffix}'.encode())
            items.append(dict(file=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                width=1280 if suffix=='mp4' else 640,height=720 if suffix=='mp4' else 360,
                frame_count=frames,fps=fps,duration_s=12.))
        records.append(dict(name=name,condition=dict(mode=mode,box_mass_kg=3.7,
            box_com_m=[.02,-.01,.03],collision=dict(time_s=3.,force_n=force)),
            variants={label:dict(controller=value[0],detector=value[1]) for label,value in variants.items()},
            display_labels={'baseline':'기준','v1':'v1','r1':'R1','r2':'R2=R1 (동일 바이트)'},
            plot_signals={'g4_support_normalized':'recorded f_table / (box_mass_kg * 9.81)'},
            timewarp=[dict(simulation_start_s=0.,simulation_end_s=6.,playback_speed=.5)],
            source_simulation_frames={label:6000 for label in V.R2_VARIANTS},rendered_frame_count=180,
            software_gl=backend,episode_summary=summaries,evaluate_classification=checks,
            reference_no_collision=references,artifacts=items))
    data=dict(schema_version=3,
        generation_command='env -u PYTHONPATH .venv/bin/python tools/render_r2_comparison.py --output-dir artifacts/media --raw-dir "$RAW_DIR" --ffmpeg ffmpeg',
        software_gl=backend,source_hashes=hashes,identical_source_aliases=aliases,
        scenarios=records,total_media_bytes=sum(x['bytes'] for row in records for x in row['artifacts']))
    (media/'r2_manifest.json').write_text(json.dumps(data));return data


class MediaValidationTests(unittest.TestCase):
    def test_valid_fixture_and_detects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            with patch.object(V,'probe',side_effect=fake_probe),patch.object(V,'read_official',side_effect=lambda root,ctrl: ORIGINAL_READ_OFFICIAL(V.ROOT,ctrl)):
                self.assertEqual(V.validate(root,'probe')['status'],'pass')
                (root/'artifacts/media/01_servo_no_collision.mp4').write_bytes(b'tampered')
                self.assertEqual(V.validate(root,'probe')['status'],'FAIL')

    def test_unexpected_codec_and_collision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=fixture(root)
            data['scenarios'][2]['condition']['collision']['force_n']=[0.,40.,0.]
            (root/'artifacts/media/media_manifest.json').write_text(json.dumps(data))
            def wrong(path,ffprobe):
                x=fake_probe(path,ffprobe)
                if path.suffix=='.mp4':x['codec']='vp9'
                return x
            with patch.object(V,'probe',side_effect=wrong),patch.object(V,'read_official',side_effect=lambda root,ctrl: ORIGINAL_READ_OFFICIAL(V.ROOT,ctrl)):
                result=V.validate(root,'probe')
            self.assertTrue(any('collision mismatch' in x for x in result['errors']))
            self.assertTrue(any('H.264' in x for x in result['errors']))

    def test_stale_reference_false_alarm_and_run_py_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=fixture(root);row=data['scenarios'][3]
            row['reference_no_collision']['baseline']['box_mass_kg']=4.
            row['run_py_equivalent']['baseline']['false_alarm_rises_in_this_episode']=7
            row['evaluate_classification']['baseline']['false_alarms']=7
            row['run_py_validation']['baseline']['parsed']['false_alarm_rises_in_this_episode']=7
            row['run_py_validation']['ours']['matches_direct_run_episode']=False
            (root/'artifacts/media/media_manifest.json').write_text(json.dumps(data))
            with patch.object(V,'probe',side_effect=fake_probe),patch.object(V,'read_official',side_effect=lambda root,ctrl: ORIGINAL_READ_OFFICIAL(V.ROOT,ctrl)):
                result=V.validate(root,'probe')
            self.assertTrue(any('reference is not matched' in x for x in result['errors']))
            self.assertTrue(any('collision false-rise count mismatch' in x for x in result['errors']))
            self.assertTrue(any('run.py confirmation missing' in x for x in result['errors']))

    def test_valid_v2_fixture_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_v2(root)
            self.assertEqual(V.validate_v2(root,'probe',fake_probe)['status'],'pass')
            (root/'my_detector.py').write_bytes(b'tampered')
            result=V.validate_v2(root,'probe',fake_probe)
            self.assertEqual(result['status'],'FAIL')
            self.assertTrue(any('source SHA mismatch' in x for x in result['errors']))

    def test_public_v2_fixture_records_absent_external_kit_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_v2(root)
            for name in V.V2_EXTERNAL_KIT_SOURCES:
                (root/name).unlink()
            result=V.validate_v2(root,'probe',fake_probe)
            self.assertEqual(result['status'],'pass',result['errors'])
            self.assertEqual(result['external_kit_sources_not_locally_verified'],
                             sorted(V.V2_EXTERNAL_KIT_SOURCES))
            data=json.loads((root/'artifacts/media/v2_manifest.json').read_text())
            data['source_hashes']['sim/runner.py']='not-a-sha256'
            (root/'artifacts/media/v2_manifest.json').write_text(json.dumps(data))
            result=V.validate_v2(root,'probe',fake_probe)
            self.assertTrue(any('sim/runner.py: source SHA mismatch' in x for x in result['errors']))

    def test_present_external_kit_source_still_requires_exact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_v2(root)
            (root/'sim/runner.py').write_bytes(b'tampered')
            result=V.validate_v2(root,'probe',fake_probe)
            self.assertTrue(any('sim/runner.py: source SHA mismatch' in x for x in result['errors']))

    def test_broken_external_source_symlink_is_not_an_absent_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_v2(root)
            source=root/'sim/runner.py';source.unlink();source.symlink_to('missing.py')
            result=V.validate_v2(root,'probe',fake_probe)
            self.assertTrue(any('sim/runner.py: source SHA mismatch' in x for x in result['errors']))
            self.assertNotIn('sim/runner.py',result['external_kit_sources_not_locally_verified'])

    def test_missing_owned_v2_source_is_not_exempted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_v2(root)
            (root/'my_detector.py').unlink()
            result=V.validate_v2(root,'probe',fake_probe)
            self.assertTrue(any('my_detector.py: source SHA mismatch' in x for x in result['errors']))

    def test_v2_rejects_8kg_gain_claim_and_wrong_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=fixture_v2(root)
            row=data['scenarios'][0]
            row['condition']['box_mass_kg']=8.
            row['variants']['v2']['controller']='invented_controller'
            (root/'artifacts/media/v2_manifest.json').write_text(json.dumps(data))
            result=V.validate_v2(root,'probe',fake_probe)
            self.assertTrue(any('physical condition mismatch' in x for x in result['errors']))
            self.assertTrue(any('module combination mismatch' in x for x in result['errors']))

    def test_valid_r2_fixture_and_alias_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=fixture_r2(root)
            self.assertEqual(V.validate_r2(root,'probe',fake_probe)['status'],'pass')
            data['identical_source_aliases'].pop('my_detector.py')
            (root/'artifacts/media/r2_manifest.json').write_text(json.dumps(data))
            result=V.validate_r2(root,'probe',fake_probe)
            self.assertTrue(any('alias declarations mismatch' in x for x in result['errors']))

    def test_r2_rejects_wrong_direction_and_missing_g4_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=fixture_r2(root)
            data['scenarios'][1]['condition']['collision']['force_n']=[0.,15.,0.]
            data['scenarios'][0]['plot_signals']={}
            (root/'artifacts/media/r2_manifest.json').write_text(json.dumps(data))
            result=V.validate_r2(root,'probe',fake_probe)
            self.assertTrue(any('physical condition mismatch' in x for x in result['errors']))
            self.assertTrue(any('G4/support-normalized' in x for x in result['errors']))

    def test_validate_all_requires_r2_only_after_r2_protocol_or_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);media=root/'artifacts/media';media.mkdir(parents=True)
            (media/'media_manifest.json').write_text('{}');(media/'v2_manifest.json').write_text('{}')
            passed={'status':'pass','errors':[]}
            with patch.object(V,'validate',return_value=passed),patch.object(V,'validate_v2',return_value=passed):
                self.assertEqual(V.validate_all(root,'probe')['status'],'pass')
                protocol=root/'results/v2_protocol/PROTOCOL_R2.json'
                protocol.parent.mkdir(parents=True);protocol.write_text('{}')
                result=V.validate_all(root,'probe')
            self.assertTrue(any('r2_manifest.json is missing' in x for x in result['errors']))

if __name__=='__main__': unittest.main()


class LegacyAliasTests(unittest.TestCase):
    def test_historical_alias_cannot_silently_target_current_v2(self):
        from tools.validate_media import legacy_alias_errors
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            aliases={}
            for stem in ['servo','torque','detector']:
                data=('fixed v1 '+stem).encode()
                (root/('v1_'+stem+'.py')).write_bytes(data)
                aliases['my_'+stem+'.py']={'file':'v1_'+stem+'.py','sha256':hashlib.sha256(data).hexdigest()}
            self.assertEqual(legacy_alias_errors(root,{'frozen_source_aliases':aliases}),[])
            aliases['my_detector.py']['file']='my_detector.py'
            self.assertTrue(legacy_alias_errors(root,{'frozen_source_aliases':aliases}))


class ArchivedRenderedSourceTests(unittest.TestCase):
    def test_unchanged_current_source_needs_no_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = b'recorded source'
            (root / 'my_detector.py').write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual('my_detector.py',
                             V.rendered_source_path(root, 'my_detector.py', digest))

    def test_changed_submission_uses_only_exact_preserved_r1_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorded = b'fixed rendered detector'
            (root / 'my_detector.py').write_bytes(b'new submission detector')
            (root / 'r1_detector.py').write_bytes(recorded)
            digest = hashlib.sha256(recorded).hexdigest()
            self.assertEqual('r1_detector.py',
                             V.rendered_source_path(root, 'my_detector.py', digest))
            (root / 'r1_detector.py').write_bytes(b'changed archive')
            self.assertIsNone(V.rendered_source_path(root, 'my_detector.py', digest))
            (root / 'r1_detector.py').unlink()
            self.assertIsNone(V.rendered_source_path(root, 'my_detector.py', digest))

    def test_archive_symlinks_and_non_submission_aliases_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorded = b'recorded'
            digest = hashlib.sha256(recorded).hexdigest()
            (root / 'my_detector.py').write_bytes(b'new')
            (root / 'target.py').write_bytes(recorded)
            (root / 'r1_detector.py').symlink_to(root / 'target.py')
            self.assertIsNone(V.rendered_source_path(root, 'my_detector.py', digest))
            (root / 'other.py').write_bytes(b'new')
            (root / 'r1_other.py').write_bytes(recorded)
            self.assertIsNone(V.rendered_source_path(root, 'other.py', digest))
            self.assertIsNone(V.rendered_source_path(root, '../other.py', digest))

    def test_v2_manifest_survives_current_alias_change_without_editing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_v2(root)
            original = (root / 'artifacts/media/v2_manifest.json').read_bytes()
            (root / 'r1_detector.py').write_bytes((root / 'my_detector.py').read_bytes())
            (root / 'my_detector.py').write_bytes(b'new submission')
            result = V.validate_v2(root, 'ffprobe', fake_probe)
            self.assertEqual('pass', result['status'], result['errors'])
            self.assertEqual({'my_detector.py': 'r1_detector.py'},
                             result['archived_render_source_aliases'])
            self.assertEqual(original, (root / 'artifacts/media/v2_manifest.json').read_bytes())

    def test_r2_manifest_survives_current_alias_change_without_editing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture_r2(root)
            original = (root / 'artifacts/media/r2_manifest.json').read_bytes()
            (root / 'my_detector.py').write_bytes(b'new submission')
            result = V.validate_r2(root, 'ffprobe', fake_probe)
            self.assertEqual('pass', result['status'], result['errors'])
            self.assertEqual({'my_detector.py': 'r1_detector.py'},
                             result['archived_render_source_aliases'])
            self.assertEqual(original, (root / 'artifacts/media/r2_manifest.json').read_bytes())
