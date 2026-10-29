"""Verify generated comparison media against its manifest and official scores.

Use an existing ffprobe; no simulator or remote connection is needed.  The
verification covers payload/hash/codec/size/timing, scenario declarations,
recorded run/evaluator decision summaries and the declared time-warp schedule.
It does not replace visual frame review or independent private run.py logs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_baseline import official_cells
STEMS = ('01_servo_no_collision', '02_torque_no_collision',
         '03_servo_place_collision', '04_servo_transport_collision')


def probe(path, executable):
    command=[str(executable),'-v','error','-count_frames','-select_streams','v:0',
             '-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames:format=duration',
             '-of','json',str(path)]
    result=subprocess.run(command,text=True,capture_output=True,timeout=120,check=False)
    if result.returncode:
        raise ValueError('ffprobe failed')
    data=json.loads(result.stdout);streams=data.get('streams',[])
    if len(streams)!=1:
        raise ValueError('expected one video stream')
    stream=streams[0]
    frames=int(stream['nb_read_frames']);duration=float(data['format']['duration'])
    if frames<=0 or duration<=0:
        raise ValueError('invalid frame count or duration')
    # GIF uses centisecond, sometimes alternating frame delays; ffprobe's
    # avg_frame_rate can reflect the first delay rather than the entire clip.
    # Count/duration gives a directly verifiable whole-clip effective rate.
    return dict(codec=stream['codec_name'],width=stream['width'],height=stream['height'],
                frame_count=frames,fps=frames/duration,duration_s=duration)


def read_official(root, ctrl):
    base='servo' if ctrl=='servo' else 'torque'
    prefix=root/'results'/f'evaluation_final_{base}'
    rows=[]
    for variant in ('base_base','final_final'):
        control,collisions=official_cells(Path(f'{prefix}_{variant}.txt').read_text(encoding='utf-8'))
        rows.append((control[8.],collisions))
    return rows


def validate(root, ffprobe):
    root=Path(root).resolve();folder=root/'artifacts/media';path=folder/'media_manifest.json'
    data=json.loads(path.read_text(encoding='utf-8'));records=data['scenarios'];errors=[]
    names=[r.get('name') for r in records]
    if sorted(names)!=sorted(STEMS): errors.append('expected exactly four uniquely named scenarios')
    seen=set();total=0;report=[]
    for row in records:
        name=row['name']; condition=row['condition'];collision=condition['collision'];files=row['artifacts']
        if condition['box_mass_kg']!=8.0 or tuple(condition['box_com_m'])!=(.02,-.01,.03):
            errors.append(f'{name}: wrong mass or COM')
        expected_time={'03_servo_place_collision':3.86,'04_servo_transport_collision':3.0}.get(name)
        if expected_time is None and collision is not None:
            errors.append(f'{name}: unexpected collision')
        elif expected_time is not None and (collision is None or collision['time_s']!=expected_time
                                            or collision['force_n']!=[0.,15.,0.]):
            errors.append(f'{name}: collision mismatch')
        if not ('llvmpipe' in row['software_gl']['renderer'].lower() or
                'softpipe' in row['software_gl']['renderer'].lower()):
            errors.append(f'{name}: not a software renderer')
        pieces=row['timewarp']
        if (not pieces or abs(pieces[0]['simulation_start_s'])>1e-9 or
            abs(pieces[-1]['simulation_end_s']-6.0)>1e-9 or
            any(p['simulation_end_s']<=p['simulation_start_s'] or p['playback_speed']<=0 for p in pieces) or
            any(abs(a['simulation_end_s']-b['simulation_start_s'])>1e-9 for a,b in zip(pieces,pieces[1:]))):
            errors.append(f'{name}: invalid timewarp schedule')
        for t in (1.4,3.9):
            if not any(p['simulation_start_s']<=t<p['simulation_end_s'] and p['playback_speed']<1 for p in pieces):
                errors.append(f'{name}: pick/place is not slowed')
        expected_duration=sum((p['simulation_end_s']-p['simulation_start_s'])/p['playback_speed'] for p in pieces)
        ctrl='torque' if name=='02_torque_no_collision' else 'servo'
        official=read_official(root,ctrl)
        summaries=row['run_py_equivalent']
        reference_info=row.get('reference_no_collision',{})
        for label,(control,_) in zip(('baseline','ours'),official):
            ref=reference_info.get(label,{})
            expected_controller=(ctrl+'_controller') if label=='baseline' else ('my_'+ctrl)
            expected_detector='collision_detector' if label=='baseline' else 'my_detector'
            if (ref.get('controller')!=expected_controller or ref.get('detector')!=expected_detector or
                ref.get('box_mass_kg')!=8.0 or tuple(ref.get('box_com_m',()))!=(.02,-.01,.03) or
                ref.get('false_alarm_rises')!=sum(map(int,control[6:9]))):
                errors.append(f'{name}: {label} no-collision reference is not matched')
        run_py_validation=row.get('run_py_validation',{})
        for label in ('baseline','ours'):
            check=run_py_validation.get(label,{})
            parsed=check.get('parsed',{})
            if not check.get('matches_direct_run_episode'):
                errors.append(f'{name}: {label} run.py confirmation missing')
            for field in ('tcp_rms_mm','tcp_max_mm','saturation_pct',
                          'false_alarm_rises_in_this_episode','alarm_before_collision','detect_delay_ms_raw'):
                if (field not in parsed or field not in summaries[label] or
                    parsed[field]!=summaries[label][field]):
                    errors.append(f'{name}: {label} run.py {field} differs from episode')
        if expected_time is None:
            for label, (control,_) in zip(('baseline','ours'),official):
                s=summaries[label]
                if (s['tcp_rms_mm'],s['tcp_max_mm'],s['saturation_pct']) != tuple(map(float,control[:3])):
                    errors.append(f'{name}: {label} differs from official no-collision control cells')
                if s['false_alarm_rises_in_this_episode']!=sum(map(int,control[6:9])):
                    errors.append(f'{name}: {label} differs from official no-collision false rises')
        else:
            checks=row['evaluate_classification']
            expected_fa=(6,0) if expected_time==3.86 else (8,0)
            for label,(_,collisions), expected_rises in zip(('baseline','ours'),official,expected_fa):
                expected=collisions[(8.,expected_time)][1]
                c=checks[label]; observed=c['collision_outcome'];delay=c['detect_delay_ms']
                labels={'놓침':'missed','경보 중':'alarm_before_collision','구분 불가':'indistinguishable'}
                if expected in labels:
                    if observed!=labels[expected]: errors.append(f'{name}: {label} official classification mismatch')
                elif observed!='detected' or delay is None or round(delay)!=int(expected):
                    errors.append(f'{name}: {label} official detection delay mismatch')
                if c['false_alarms']!=summaries[label]['false_alarm_rises_in_this_episode'] or c['false_alarms']!=expected_rises:
                    errors.append(f'{name}: {label} collision false-rise count mismatch')
                if (expected_time==3.86 and label=='baseline' and
                    (not summaries[label]['alarm_before_collision'] or summaries[label]['detect_delay_ms_raw'] is not None)):
                    errors.append(f'{name}: baseline must be alarm-before with no raw delay')
        if len(files)!=2:
            errors.append(f'{name}: expected MP4 and GIF')
        for item in files:
            relative=item['file'];f=root/relative
            if relative not in {f'artifacts/media/{name}.mp4',f'artifacts/media/{name}.gif'} or relative in seen:
                errors.append(f'{name}: invalid or duplicate media path');continue
            seen.add(relative)
            if not f.is_file() or f.is_symlink():
                errors.append(f'{relative}: missing or unsafe');continue
            raw=f.read_bytes();total+=len(raw)
            if item['bytes']!=len(raw) or item['sha256']!=hashlib.sha256(raw).hexdigest():
                errors.append(f'{relative}: bytes/SHA mismatch')
            try:actual=probe(f,ffprobe)
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired,json.JSONDecodeError):
                errors.append(f'{relative}: probe failed');continue
            for field in ('width','height','frame_count'):
                if item[field]!=actual[field]: errors.append(f'{relative}: {field} mismatch')
            if abs(item['fps']-actual['fps'])>0.01 or abs(item['duration_s']-actual['duration_s'])>0.03:
                errors.append(f'{relative}: fps/duration mismatch')
            if relative.endswith('.mp4'):
                if actual['codec']!='h264' or (actual['width'],actual['height'])!=(1280,720):
                    errors.append(f'{relative}: expected H.264 1280x720')
                if item['frame_count']!=row['rendered_frame_count']:
                    errors.append(f'{relative}: rendered frame count mismatch')
                if abs(actual['duration_s']-expected_duration) > (len(pieces)+2)/actual['fps']:
                    errors.append(f'{relative}: duration inconsistent with timewarp schedule')
            elif actual['codec']!='gif' or len(raw)>6_000_000:
                errors.append(f'{relative}: expected GIF below 6 MB')
            report.append(dict(file=relative,bytes=len(raw),sha256=item['sha256'],**actual))
    if len(seen)!=8 or total!=data['total_media_bytes'] or total>30_000_000:
        errors.append('media total/file coverage mismatch')
    return dict(status='pass' if not errors else 'FAIL', errors=errors, files=report,
                total_media_bytes=total, scenarios=len(records),
                scope='manifest, ffprobe, codec, timewarp and official summary; independent visual/run.py verification also required')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--ffprobe',required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(); result=validate(args.root,args.ffprobe)
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.out: args.out.write_text(text,encoding='utf-8')
    print(text,end='')
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':
    raise SystemExit(main())
