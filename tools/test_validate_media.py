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

if __name__=='__main__': unittest.main()
