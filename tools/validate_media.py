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
V2_STEMS = ('05_v2_servo_transport_collision',
            '06_v2_torque_transport_collision')
V2_VARIANTS = ('baseline', 'v1', 'v2')
R2_STEMS = ('07_r2_servo_comparison', '08_r2_torque_comparison')
R2_VARIANTS = ('baseline', 'v1', 'r1', 'r2')
V2_EXTERNAL_KIT_SOURCES = {
    'servo_controller.py', 'torque_controller.py', 'collision_detector.py', 'sim/runner.py',
}


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


def legacy_alias_errors(root, data):
    aliases = data.get('frozen_source_aliases')
    if aliases is None:
        return []
    names = {'my_servo.py','my_torque.py','my_detector.py'}
    if set(aliases) != names:
        return ['legacy v1 source alias coverage mismatch']
    errors = []
    for name, item in aliases.items():
        expected = 'v1_'+name[3:]
        path = Path(root)/expected
        if (item.get('file') != expected or path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != item.get('sha256')):
            errors.append(f'{name}: frozen v1 alias/hash mismatch')
    return errors


def validate(root, ffprobe):
    root=Path(root).resolve();folder=root/'artifacts/media';path=folder/'media_manifest.json'
    data=json.loads(path.read_text(encoding='utf-8'));records=data['scenarios'];errors=[]
    errors.extend(legacy_alias_errors(root, data))
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


def _v2_combinations(mode):
    return {
        'baseline': (f'{mode}_controller', 'collision_detector'),
        'v1': (f'v1_{mode}', 'v1_detector'),
        'v2': (f'my_{mode}', 'my_detector'),
    }


def _r2_combinations(mode):
    return {
        'baseline': (f'{mode}_controller', 'collision_detector'),
        'v1': (f'v1_{mode}', 'v1_detector'),
        'r1': (f'r1_{mode}', 'r1_detector'),
        'r2': (f'my_{mode}', 'my_detector'),
    }


def validate_v2(root, ffprobe, probe_fn=probe):
    """Validate the distinct baseline/v1/v2 media set without simulation."""
    root=Path(root).resolve();folder=root/'artifacts/media';path=folder/'v2_manifest.json'
    data=json.loads(path.read_text(encoding='utf-8'));records=data.get('scenarios',[]);errors=[]
    names=[row.get('name') for row in records]
    if sorted(names)!=sorted(V2_STEMS) or len(set(names))!=len(V2_STEMS):
        errors.append('expected exactly two uniquely named v2 scenarios')
    command=data.get('generation_command','')
    forbidden=('/'+'home/','TASK-','Notion','materials')
    if ('tools/render_v2_comparison.py' not in command or
            '--output-dir artifacts/media' not in command or '--raw-dir "$RAW_DIR"' not in command or
            '--ffmpeg ffmpeg' not in command or any(x in command for x in forbidden)):
        errors.append('v2 generation command is not portable')
    backend=data.get('software_gl',{})
    identity=(backend.get('vendor','')+' '+backend.get('renderer','')).lower()
    if 'mesa' not in identity or not ('llvmpipe' in identity or 'softpipe' in identity):
        errors.append('v2 manifest does not prove actual Mesa software GL')
    hashes=data.get('source_hashes',{})
    required_sources={
        'servo_controller.py','torque_controller.py','collision_detector.py',
        'v1_servo.py','v1_torque.py','v1_detector.py',
        'my_servo.py','my_torque.py','my_detector.py','sim/runner.py',
        'tools/render_v2_comparison.py',
    }
    if set(hashes)!=required_sources:
        errors.append('v2 source hash coverage mismatch')
    external_not_local=[]
    for relative,digest in hashes.items():
        source=root/relative
        safe_digest=isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not safe_digest:
            errors.append(f'{relative}: source SHA mismatch')
        elif source.is_symlink() or (source.exists() and not source.is_file()):
            errors.append(f'{relative}: source SHA mismatch')
        elif relative in V2_EXTERNAL_KIT_SOURCES and not source.exists():
            external_not_local.append(relative)
        elif not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            errors.append(f'{relative}: source SHA mismatch')
    if (hashes.get('v1_servo.py')!=hashes.get('my_servo.py') or
            hashes.get('v1_torque.py')!=hashes.get('my_torque.py')):
        errors.append('v1/v2 controller sources must remain identical')
    if hashes.get('v1_detector.py')==hashes.get('my_detector.py'):
        errors.append('v2 detector source is not distinct from v1')

    seen=set();total=0;report=[]
    for row in records:
        name=row.get('name','');condition=row.get('condition',{});mode=condition.get('mode')
        collision=condition.get('collision');expected_variants=_v2_combinations(mode) if mode in ('servo','torque') else {}
        expected_name=(f"{'05' if mode=='servo' else '06'}_v2_{mode}_transport_collision"
                       if mode in ('servo','torque') else None)
        if name!=expected_name:
            errors.append(f'{name}: mode/stem mismatch')
        if (condition.get('box_mass_kg')!=3.7 or tuple(condition.get('box_com_m',()))!=(.02,-.01,.03) or
                not isinstance(collision,dict) or collision.get('time_s')!=3.0 or
                collision.get('force_n')!=[0.,15.,0.]):
            errors.append(f'{name}: v2 physical condition mismatch')
        variants=row.get('variants',{})
        if set(variants)!=set(V2_VARIANTS):
            errors.append(f'{name}: expected baseline/v1/v2 variants')
        for label,(controller,detector) in expected_variants.items():
            if variants.get(label)!={'controller':controller,'detector':detector}:
                errors.append(f'{name}: {label} module combination mismatch')
        row_backend=row.get('software_gl',{})
        if row_backend!=backend:
            errors.append(f'{name}: software GL identity differs from manifest')
        pieces=row.get('timewarp',[])
        if (not pieces or abs(pieces[0].get('simulation_start_s',-1))>1e-9 or
                abs(pieces[-1].get('simulation_end_s',-1)-6.0)>1e-9 or
                any(p.get('simulation_end_s',0)<=p.get('simulation_start_s',0) or
                    p.get('playback_speed',0)<=0 for p in pieces) or
                any(abs(a['simulation_end_s']-b['simulation_start_s'])>1e-9
                    for a,b in zip(pieces,pieces[1:])) or
                not any(p['simulation_start_s']<=3.0<p['simulation_end_s'] and
                        p['playback_speed']<1 for p in pieces)):
            errors.append(f'{name}: invalid v2 timewarp schedule')
        summaries=row.get('episode_summary',{});checks=row.get('evaluate_classification',{})
        references=row.get('reference_no_collision',{});source_frames=row.get('source_simulation_frames',{})
        if any(set(group)!=set(V2_VARIANTS) for group in (summaries,checks,references,source_frames)):
            errors.append(f'{name}: incomplete three-way episode evidence')
        for label,(controller,detector) in expected_variants.items():
            summary=summaries.get(label,{});check=checks.get(label,{});reference=references.get(label,{})
            if (reference.get('controller')!=controller or reference.get('detector')!=detector or
                    reference.get('box_mass_kg')!=3.7 or
                    tuple(reference.get('box_com_m',()))!=(.02,-.01,.03) or
                    not isinstance(reference.get('resource_guard_before_reference'),dict) or
                    not isinstance(reference.get('resource_guard_before_collision'),dict)):
                errors.append(f'{name}: {label} reference/resource guard mismatch')
            outcome=check.get('collision_outcome');delay=check.get('detect_delay_ms')
            if outcome not in ('detected','missed','alarm_before_collision','indistinguishable'):
                errors.append(f'{name}: {label} invalid official classification')
            summary_delay=summary.get('detect_delay_ms_raw')
            delay_matches=(delay is None and summary_delay is None) or (
                delay is not None and summary_delay is not None and abs(delay-summary_delay)<=.51)
            if (check.get('false_alarms')!=summary.get('false_alarm_rises_in_this_episode') or
                    bool(check.get('alarm_before_collision'))!=bool(summary.get('alarm_before_collision')) or
                    not delay_matches):
                errors.append(f'{name}: {label} classification/episode mismatch')
            if not isinstance(source_frames.get(label),int) or source_frames.get(label)<=0:
                errors.append(f'{name}: {label} invalid source frame count')
        files=row.get('artifacts',[])
        if len(files)!=2:
            errors.append(f'{name}: expected v2 MP4 and GIF')
        expected_duration=sum((p['simulation_end_s']-p['simulation_start_s'])/p['playback_speed']
                              for p in pieces) if pieces else 0
        for item in files:
            relative=item.get('file','');media=root/relative
            expected={f'artifacts/media/{name}.mp4',f'artifacts/media/{name}.gif'}
            if relative not in expected or relative in seen:
                errors.append(f'{name}: invalid or duplicate v2 media path');continue
            seen.add(relative)
            if not media.is_file() or media.is_symlink():
                errors.append(f'{relative}: missing or unsafe');continue
            raw=media.read_bytes();total+=len(raw)
            if item.get('bytes')!=len(raw) or item.get('sha256')!=hashlib.sha256(raw).hexdigest():
                errors.append(f'{relative}: bytes/SHA mismatch')
            try: actual=probe_fn(media,ffprobe)
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired,json.JSONDecodeError):
                errors.append(f'{relative}: probe failed');continue
            for field in ('width','height','frame_count'):
                if item.get(field)!=actual[field]: errors.append(f'{relative}: {field} mismatch')
            if abs(item.get('fps',0)-actual['fps'])>.01 or abs(item.get('duration_s',0)-actual['duration_s'])>.03:
                errors.append(f'{relative}: fps/duration mismatch')
            if relative.endswith('.mp4'):
                if actual['codec']!='h264' or (actual['width'],actual['height'])!=(1280,720) or abs(actual['fps']-15)>0.05:
                    errors.append(f'{relative}: expected H.264 1280x720 at 15 fps')
                if item.get('frame_count')!=row.get('rendered_frame_count'):
                    errors.append(f'{relative}: rendered frame count mismatch')
                if expected_duration and abs(actual['duration_s']-expected_duration)>(len(pieces)+2)/actual['fps']:
                    errors.append(f'{relative}: duration inconsistent with timewarp schedule')
            elif actual['codec']!='gif' or (actual['width'],actual['height'])!=(640,360) or abs(actual['fps']-8)>.15:
                errors.append(f'{relative}: expected 640x360 GIF at 8 fps')
            report.append(dict(file=relative,bytes=len(raw),sha256=item.get('sha256'),**actual))
    if len(seen)!=4 or total!=data.get('total_media_bytes') or total>30_000_000:
        errors.append('v2 media total/file coverage mismatch')
    return dict(status='pass' if not errors else 'FAIL',errors=errors,files=report,
                total_media_bytes=total,scenarios=len(records),
                external_kit_sources_not_locally_verified=sorted(external_not_local),
                scope='v2 manifest, sources, software GL, three variants, classification, ffprobe and hashes')


def validate_r2(root, ffprobe, probe_fn=probe):
    """Validate four-way R2 media and its byte-identity declarations."""
    root=Path(root).resolve();folder=root/'artifacts/media';path=folder/'r2_manifest.json'
    data=json.loads(path.read_text(encoding='utf-8'));records=data.get('scenarios',[]);errors=[]
    names=[row.get('name') for row in records]
    if sorted(names)!=sorted(R2_STEMS) or len(set(names))!=len(R2_STEMS):
        errors.append('expected exactly two uniquely named R2 scenarios')
    command=data.get('generation_command','')
    forbidden=('/'+'home/','TASK-','Notion','materials')
    if ('tools/render_r2_comparison.py' not in command or
            '--output-dir artifacts/media' not in command or '--raw-dir "$RAW_DIR"' not in command or
            '--ffmpeg ffmpeg' not in command or any(x in command for x in forbidden)):
        errors.append('R2 generation command is not portable')
    backend=data.get('software_gl',{})
    identity=(backend.get('vendor','')+' '+backend.get('renderer','')).lower()
    if 'mesa' not in identity or not ('llvmpipe' in identity or 'softpipe' in identity):
        errors.append('R2 manifest does not prove actual Mesa software GL')
    hashes=data.get('source_hashes',{})
    required_sources={
        'servo_controller.py','torque_controller.py','collision_detector.py',
        'v1_servo.py','v1_torque.py','v1_detector.py',
        'r1_servo.py','r1_torque.py','r1_detector.py',
        'my_servo.py','my_torque.py','my_detector.py','sim/runner.py',
        'tools/render_r2_comparison.py',
    }
    if set(hashes)!=required_sources:
        errors.append('R2 source hash coverage mismatch')
    external_not_local=[]
    for relative,digest in hashes.items():
        source=root/relative
        safe_digest=isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not safe_digest:
            errors.append(f'{relative}: source SHA mismatch')
        elif source.is_symlink() or (source.exists() and not source.is_file()):
            errors.append(f'{relative}: source SHA mismatch')
        elif relative in V2_EXTERNAL_KIT_SOURCES and not source.exists():
            external_not_local.append(relative)
        elif not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            errors.append(f'{relative}: source SHA mismatch')
    pairs=(('my_servo.py','r1_servo.py'),('my_torque.py','r1_torque.py'),
           ('my_detector.py','r1_detector.py'))
    expected_aliases={new:old for new,old in pairs if hashes.get(new)==hashes.get(old)}
    if data.get('identical_source_aliases')!=expected_aliases:
        errors.append('R2 identical-source alias declarations mismatch')

    seen=set();total=0;report=[]
    expected_conditions={
        'servo': ('07_r2_servo_comparison',[0.,15.,0.]),
        'torque': ('08_r2_torque_comparison',[-15.,0.,0.]),
    }
    for row in records:
        name=row.get('name','');condition=row.get('condition',{});mode=condition.get('mode')
        collision=condition.get('collision');expected_variants=_r2_combinations(mode) if mode in expected_conditions else {}
        expected_name,expected_force=expected_conditions.get(mode,(None,None))
        if name!=expected_name:
            errors.append(f'{name}: R2 mode/stem mismatch')
        if (condition.get('box_mass_kg')!=3.7 or tuple(condition.get('box_com_m',()))!=(.02,-.01,.03) or
                not isinstance(collision,dict) or collision.get('time_s')!=3.0 or
                collision.get('force_n')!=expected_force):
            errors.append(f'{name}: R2 physical condition mismatch')
        variants=row.get('variants',{})
        if set(variants)!=set(R2_VARIANTS):
            errors.append(f'{name}: expected baseline/v1/R1/R2 variants')
        for label,(controller,detector) in expected_variants.items():
            if variants.get(label)!={'controller':controller,'detector':detector}:
                errors.append(f'{name}: {label} module combination mismatch')
        labels=row.get('display_labels',{})
        r1_r2_equal=(hashes.get(f'my_{mode}.py')==hashes.get(f'r1_{mode}.py') and
                     hashes.get('my_detector.py')==hashes.get('r1_detector.py'))
        if set(labels)!=set(R2_VARIANTS) or (('R2=R1' in labels.get('r2','')) != r1_r2_equal):
            errors.append(f'{name}: R2 identical-byte display label mismatch')
        plot_signals=row.get('plot_signals',{})
        if plot_signals.get('g4_support_normalized')!='recorded f_table / (box_mass_kg * 9.81)':
            errors.append(f'{name}: G4/support-normalized trace declaration missing')
        if row.get('software_gl',{})!=backend:
            errors.append(f'{name}: software GL identity differs from R2 manifest')
        pieces=row.get('timewarp',[])
        if (not pieces or abs(pieces[0].get('simulation_start_s',-1))>1e-9 or
                abs(pieces[-1].get('simulation_end_s',-1)-6.0)>1e-9 or
                any(p.get('simulation_end_s',0)<=p.get('simulation_start_s',0) or
                    p.get('playback_speed',0)<=0 for p in pieces) or
                any(abs(a['simulation_end_s']-b['simulation_start_s'])>1e-9
                    for a,b in zip(pieces,pieces[1:])) or
                not any(p['simulation_start_s']<=3.0<p['simulation_end_s'] and
                        p['playback_speed']<1 for p in pieces)):
            errors.append(f'{name}: invalid R2 timewarp schedule')
        summaries=row.get('episode_summary',{});checks=row.get('evaluate_classification',{})
        references=row.get('reference_no_collision',{});source_frames=row.get('source_simulation_frames',{})
        if any(set(group)!=set(R2_VARIANTS) for group in (summaries,checks,references,source_frames)):
            errors.append(f'{name}: incomplete four-way episode evidence')
        if r1_r2_equal and (summaries.get('r1')!=summaries.get('r2') or
                            checks.get('r1')!=checks.get('r2')):
            errors.append(f'{name}: byte-identical R1/R2 outcomes differ')
        for label,(controller,detector) in expected_variants.items():
            summary=summaries.get(label,{});check=checks.get(label,{});reference=references.get(label,{})
            if (reference.get('controller')!=controller or reference.get('detector')!=detector or
                    reference.get('box_mass_kg')!=3.7 or tuple(reference.get('box_com_m',()))!=(.02,-.01,.03) or
                    not isinstance(reference.get('resource_guard_before_reference'),dict) or
                    not isinstance(reference.get('resource_guard_before_collision'),dict)):
                errors.append(f'{name}: {label} reference/resource guard mismatch')
            outcome=check.get('collision_outcome');delay=check.get('detect_delay_ms')
            if outcome not in ('detected','missed','alarm_before_collision','indistinguishable'):
                errors.append(f'{name}: {label} invalid classification')
            summary_delay=summary.get('detect_delay_ms_raw')
            delay_matches=(delay is None and summary_delay is None) or (
                delay is not None and summary_delay is not None and abs(delay-summary_delay)<=.51)
            if (check.get('false_alarms')!=summary.get('false_alarm_rises_in_this_episode') or
                    bool(check.get('alarm_before_collision'))!=bool(summary.get('alarm_before_collision')) or
                    not delay_matches):
                errors.append(f'{name}: {label} classification/episode mismatch')
            if not isinstance(source_frames.get(label),int) or source_frames.get(label)<=0:
                errors.append(f'{name}: {label} invalid source frame count')
        files=row.get('artifacts',[])
        if len(files)!=2:
            errors.append(f'{name}: expected R2 MP4 and GIF')
        expected_duration=sum((p['simulation_end_s']-p['simulation_start_s'])/p['playback_speed']
                              for p in pieces) if pieces else 0
        for item in files:
            relative=item.get('file','');media=root/relative
            expected={f'artifacts/media/{name}.mp4',f'artifacts/media/{name}.gif'}
            if relative not in expected or relative in seen:
                errors.append(f'{name}: invalid or duplicate R2 media path');continue
            seen.add(relative)
            if not media.is_file() or media.is_symlink():
                errors.append(f'{relative}: missing or unsafe');continue
            raw=media.read_bytes();total+=len(raw)
            if item.get('bytes')!=len(raw) or item.get('sha256')!=hashlib.sha256(raw).hexdigest():
                errors.append(f'{relative}: bytes/SHA mismatch')
            try: actual=probe_fn(media,ffprobe)
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired,json.JSONDecodeError):
                errors.append(f'{relative}: probe failed');continue
            for field in ('width','height','frame_count'):
                if item.get(field)!=actual[field]: errors.append(f'{relative}: {field} mismatch')
            if abs(item.get('fps',0)-actual['fps'])>.01 or abs(item.get('duration_s',0)-actual['duration_s'])>.03:
                errors.append(f'{relative}: fps/duration mismatch')
            if relative.endswith('.mp4'):
                if actual['codec']!='h264' or (actual['width'],actual['height'])!=(1280,720) or abs(actual['fps']-15)>.05:
                    errors.append(f'{relative}: expected H.264 1280x720 at 15 fps')
                if item.get('frame_count')!=row.get('rendered_frame_count'):
                    errors.append(f'{relative}: rendered frame count mismatch')
                if expected_duration and abs(actual['duration_s']-expected_duration)>(len(pieces)+2)/actual['fps']:
                    errors.append(f'{relative}: duration inconsistent with timewarp schedule')
            elif actual['codec']!='gif' or (actual['width'],actual['height'])!=(640,360) or abs(actual['fps']-8)>.15:
                errors.append(f'{relative}: expected 640x360 GIF at 8 fps')
            report.append(dict(file=relative,bytes=len(raw),sha256=item.get('sha256'),**actual))
    if len(seen)!=4 or total!=data.get('total_media_bytes') or total>30_000_000:
        errors.append('R2 media total/file coverage mismatch')
    return dict(status='pass' if not errors else 'FAIL',errors=errors,files=report,
                total_media_bytes=total,scenarios=len(records),
                external_kit_sources_not_locally_verified=sorted(external_not_local),
                scope='R2 manifest, sources, aliases, software GL, four variants, G4 trace, classification, ffprobe and hashes')


def validate_all(root, ffprobe):
    """Validate every media manifest present; never silently skip v2."""
    root=Path(root).resolve();media=root/'artifacts/media';sets={};errors=[]
    legacy=media/'media_manifest.json';v2=media/'v2_manifest.json';r2=media/'r2_manifest.json'
    if legacy.exists():
        sets['legacy']=validate(root,ffprobe)
    else:
        errors.append('legacy media_manifest.json is missing')
    if v2.exists():
        sets['v2']=validate_v2(root,ffprobe)
    else:
        errors.append('v2 v2_manifest.json is missing')
    r2_required=(root/'results/v2_protocol/PROTOCOL_R2.json').exists() or r2.exists()
    if r2.exists():
        sets['r2']=validate_r2(root,ffprobe)
    elif r2_required:
        errors.append('R2 r2_manifest.json is missing')
    for label,result in sets.items():
        errors.extend(f'{label}: {error}' for error in result['errors'])
    return dict(status='pass' if not errors else 'FAIL',errors=errors,sets=sets,
                validated_manifests=sorted(sets))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--ffprobe',default='ffprobe')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(); result=validate_all(args.root,args.ffprobe)
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.out: args.out.write_text(text,encoding='utf-8')
    print(text,end='')
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':
    raise SystemExit(main())
