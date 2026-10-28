"""Build final tables, failure-map figures and presentation from saved results.

No simulation and no network operations. Official and supplemental data remain
explicitly separated. The three production files are not modified.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.d4_robustness import DIRECTIONS, FORCES
from tools.compare_baseline import official_cells

R = ROOT/'results'


def load(name):
    return json.loads((R/name).read_text())


def table(combinations):
    lines = ['| Combination (8 kg) | RMS / max mm | Pick / place N m | Table N | Sat % | False pick/place/other |',
             '|---|---:|---:|---:|---:|---|']
    for name, combo in combinations.items():
        row = combo['control']['8.0']
        lines.append(f"| {name} | {row['rms_mm']:.2f} / {row['max_mm']:.2f} | "
                     f"{row['pick_nm']:.1f} / {row['place_nm']:.1f} | {row['table_n']:.0f} | "
                     f"{row['saturation_pct']:g} | {'/'.join(map(str,row['false_alarms']))} |")
    return lines


def figures(official, robust, timing, shared):
    out = R/'figures'
    out.mkdir(exist_ok=True)
    names = ['servo_base_base', 'servo_final_final', 'torque_base_base', 'torque_final_final']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fields = [('rms_mm','TCP RMS [mm]'),('max_mm','TCP max [mm]'),('pick_nm','Pick step [N m]'),
              ('place_nm','Place step [N m]'),('table_n','Place table [N]'),('false_alarms','False alarm rises')]
    for ax, (field,title) in zip(axes.flat, fields):
        values = [official[n]['control']['8.0'][field] for n in names]
        if field == 'false_alarms': values = [sum(v) for v in values]
        ax.bar(['A base','A ours','B base','B ours'], values,
               color=['#78909c','#00796b','#bcaaa4','#d16b28'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Same-PC official evaluation, 8 kg; all four have zero saturation')
    fig.tight_layout()
    fig.savefig(out/'official_comparison.png', dpi=160)
    plt.close(fig)
    codes = {'detected':0, 'missed':1, 'alarm_before_collision':2, 'indistinguishable':3}
    for mode, result in robust.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
        for ax, phase in zip(axes, ('pick','transport','place')):
            array = np.array([[codes[result['failure_map'][f'{phase}/{d}/{f:g}N']['outcome']]
                               for f in FORCES] for d in DIRECTIONS])
            image = ax.imshow(array, aspect='auto', vmin=0, vmax=3,
                              cmap=ListedColormap(['#4caf8b','#ed6a5e','#c69c35','#8c77a9']))
            ax.set_xticks(range(len(FORCES)), [f'{f:g}' for f in FORCES])
            ax.set_yticks(range(len(DIRECTIONS)), DIRECTIONS)
            ax.set_xlabel('Force norm [N]')
            ax.set_title(phase)
            for y in range(len(DIRECTIONS)):
                for x in range(len(FORCES)):
                    item = result['failure_map'][f'{phase}/{list(DIRECTIONS)[y]}/{FORCES[x]:g}N']
                    label = str(round(item['delay_ms'])) if item['outcome']=='detected' else ['D','M','A','I'][array[y,x]]
                    ax.text(x,y,label,ha='center',va='center',fontsize=8)
        fig.suptitle(f'{mode}: stratified failure map; D/delay=detected, M=miss, A=prior, I=indistinguishable\n'
                     'Mass, COM and time vary across cells: not fixed-condition force thresholds')
        fig.tight_layout()
        fig.savefig(out/f'failure_map_{mode}.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5))
    for row in timing['rows']:
        c = row['components']['combined']
        edges, counts = np.asarray(c['histogram_edges_us']), np.asarray(c['histogram_counts'])
        ax.step(edges[1:], np.cumsum(counts)/sum(counts), where='post',
                label=f"{row['controller']} {row['mass_kg']:g} kg: p99 {c['p99_us']:.0f}, max {c['max_us']:.0f} us")
    ax.set_xlim(0,1200)
    ax.axvline(1000,color='red',ls='--')
    ax.set_xlabel('Compute + update wall time [us]')
    ax.set_ylabel('Empirical CDF (10 us histogram, full tails in JSON)')
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'timing_distribution.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax,mode in zip(axes,('servo','torque')):
        rows = [r for r in shared['rows'] if r['mode']==mode and r['blend']==.1]
        rows.sort(key=lambda r:r['shared'])
        ax.bar(['independent','shared'],[r['timing']['combined']['median_us'] for r in rows], color=['#78909c','#00796b'])
        ax.set_title(mode+': identical physical/decision output')
        ax.set_ylabel('Compute + update median [us]')
        ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'sharing_contribution.png',dpi=160);plt.close(fig)


def assessment(official, robust, timing):
    verdict = {}
    for mode in ('servo','torque'):
        base = official[mode+'_base_base']['control']['8.0']
        row = official[mode+'_final_final']['control']['8.0']
        tests = dict(pick_reduction_75pct=row['pick_nm']<=.25*base['pick_nm'],
                     place_reduction_75pct=row['place_nm']<=.25*base['place_nm'],
                     table_le_1p1mg=row['table_n']<=1.1*8*9.81,
                     tcp_max_le_baseline=row['max_mm']<=base['max_mm'],
                     zero_saturation=row['saturation_pct']==0,
                     zero_false_alarms_all_official_masses=all(sum(r['false_alarms'])==0
                         for r in official[mode+'_final_final']['control'].values()))
        cols = [c for c in official[mode+'_final_final']['collisions'] if c['mass']==8]
        transport = next(c for c in cols if c['time']==3.)
        delay_comparisons=[]
        for base_collision in official[mode+'_base_base']['collisions']:
            if base_collision['mass']!=8:continue
            final_collision=next(c for c in cols if c['time']==base_collision['time'])
            for force in ('10N','15N','40N','100N'):
                try: base_delay=float(base_collision[force])
                except (ValueError,TypeError):continue
                try: final_delay=float(final_collision[force])
                except (ValueError,TypeError):final_delay=None
                delay_comparisons.append(dict(time_s=base_collision['time'],force=force,
                    baseline_ms=base_delay,final_ms=final_delay,
                    no_regression=final_delay is not None and 0<=final_delay<=base_delay))
        strict_max = max(robust[mode]['timing']['combined']['observed_max_us'],
                         *(r['components']['combined']['max_us'] for r in timing['rows']
                           if r['controller']=='my_'+mode))
        tests.update(no_prior_or_indistinguishable=all(c[f] not in ('경보 중','구분 불가')
                               for c in cols for f in ('10N','15N','40N','100N')),
                     transport_minimum_below_15n=transport['10N'] not in ('놓침','경보 중','구분 불가'),
                     same_pc_delay_no_regression=bool(delay_comparisons) and all(c['no_regression'] for c in delay_comparisons),
                     strict_observed_every_cycle_below_1ms=strict_max<1000.)
        verdict[mode] = dict(tests=tests, observed_max_us=strict_max,
                            delay_comparisons=delay_comparisons,
                            pick_reduction_pct=100*(1-row['pick_nm']/base['pick_nm']),
                            place_reduction_pct=100*(1-row['place_nm']/base['place_nm']),
                            sustained_pre_release_handover=False,
                            overall_acceptance=False)
    return verdict


def make_slides(official, robust, goals):
    path = R/'presentation.pdf'
    with PdfPages(path) as pdf:
        def slide(title, lines=(), image=None):
            fig = plt.figure(figsize=(13.33,7.5), facecolor='#f8fafb')
            fig.text(.055,.91,title,fontsize=23,weight='bold',color='#12384a')
            if image is not None:
                ax=fig.add_axes([.04,.07,.92,.78]);ax.imshow(plt.imread(R/'figures'/image));ax.axis('off')
            else:
                y=.79
                for line in lines:
                    wrapped = textwrap.wrap(line,100) or ['']
                    for text in wrapped:
                        fig.text(.065,y,text,fontsize=16,color='#172f3b');y-=.053
                    y-=.018
            fig.text(.055,.025,'Simulation study | fixed 6 s cycle | official scores separated from analysis',fontsize=9,color='#647781')
            pdf.savefig(fig);plt.close(fig)
        slide('Load transfer is not a binary payload switch',[
            'Individual advanced manipulation study: M1013 palletizing, A servo vs B torque.',
            'Result: elastic feedforward + continuous load transition reduce torque steps and TCP error.',
            'Wrench-consistent classification removes official no-collision false alarms.',
            'Not solved: <15 N transport sensitivity, sustained pre-release unloading, universal robustness or hard real time.'])
        slide('Problem definition: baseline A','', 'control_servo.png')
        slide('Idea 1: elastic motor reference and feedforward',[
            'theta_d = q_d + K^-1 tau_J,d;  tau_J,d = tau_0 + lambda_hat (tau_box - tau_0).',
            'tau_ff = B theta_ddot + tau_J,d + viscous + tanh Coulomb friction.',
            'One-sample trajectory advance; bounded/filtered elastic derivatives.',
            'A retains the fixed cascaded drive. B adds low-bandwidth motor PD and small JTS torque feedback.',
            'B2 uses an inertia-shaping interpretation, not a demonstrated sampled-data passivity theorem.'])
        slide('Idea 2: table-consistent wrench and blind thresholds',[
            'p = M(qhat) vhat; r = K_O [p - integral(tau_s - bias_hat - beta + r)dt].',
            'w_hat = J_bottom^-T r; subtract feasible upward support with COP inside the rotated footprint.',
            'Support allowed only with grip, positive box mass and taught table-height band.',
            'Residual horizontal/negative-normal/COP/torsion/out-of-height evidence + grip-off transient channel.',
            'Symmetric amplitudes and persistence fitted on no-collision training only; disjoint holdout unused for fitting.'])
        slide('Idea 3: shared estimator, separate contributions',[
            'Standalone files embed identical estimator code; optional self.ctrl connection, no runtime helper import.',
            'Detector executes first; timestamp caching gives one estimator advance per sample.',
            'Same physical and detection outputs: shared 6000 vs independent 12000 advances.',
            '10% bounded support fusion is a separate algorithmic contribution; sharing itself saves compute only.',
            'Aggressive early-settle candidates were rejected. Sustained handover before release remains unmet.'])
        slide('Official evaluation: baseline vs final','','official_comparison.png')
        slide('Ablation: changes that helped and changes that did not',[
            'A baseline: 6.14/14.00 mm RMS/max; pick/place step 57.5/73.8 N m.',
            'C1 alone: 1.17/3.37 mm but 107.7/106.6 N m. Elastic tracking alone worsens edge steps.',
            'C1+C2: 1.20/3.45 mm, 9.7/17.7 N m. Continuous load FF is the main step reduction.',
            'B baseline: 7.04/13.18 mm, 57.2/74.8 N m. B1 alone: 2.24/4.45 mm, 97.9/96.7 N m.',
            'V reduces pickup Fz peak but not overall oracle RMSE; bias/wrench logic and blind thresholds remove false alarms.',
            'Historical stages, final support fusion and computation reuse are separated in the result tables.'])
        slide('A transfer detail: original-reference error cost','','transfer_servo.png')
        slide('B transfer detail: freedom is not free robustness','','transfer_torque.png')
        slide('Collision detection: estimated external force','','detection_transport.png')
        slide('A finite-condition failure map','','failure_map_servo.png')
        slide('B finite-condition failure map','','failure_map_torque.png')
        slide('Computation: distributions, not a hard RT claim','','timing_distribution.png')
        slide('Reuse contribution: numerically identical decisions','','sharing_contribution.png')
        slide('Utility, scope and costs',[
            'Elastic feedforward is useful independently of transfer estimation; continuous payload FF greatly reduces switching steps.',
            'A gives better tracking and less friction sensitivity here; B gives direct torque authority but more model sensitivity.',
            '770 stratified nominal episodes + 50 analysis-only perturbations are finite coverage, not probabilities.',
            'Vertical force on a feasible table-normal line is indistinguishable from support with these signals.',
            'No hardware validation, collision-triggered stop or safety certification. Additional sensing/process design is needed.'])
        slide('Reproduction and primary references',[
            'Use the separately supplied unchanged course kit; our public package excludes its code/models.',
            'README and report tables include commands, software versions and artifact checksums.',
            'De Luca & Mattone, ICRA 2005: generalized-momentum collision residual.',
            'Albu-Schaeffer, Ott & Hirzinger, IJRR 2007: flexible-joint torque feedback and inertia shaping.',
            'MuJoCo official computation documentation; exact links and caveats in docs/REFERENCES.md.'])


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    official=load('FINAL_OFFICIAL.json')['combinations']
    robust={mode:load('D4_ROBUST_my_'+mode+'.json') for mode in ('servo','torque')}
    timing=load('FINAL_TIMING_DISTRIBUTION.json')
    shared=load('D4_SHARED_ABLATION.json')
    goals=assessment(official,robust,timing)
    (R/'FINAL_GOALS.json').write_text(json.dumps(goals,indent=2)+'\n')
    figures(official,robust,timing,shared)
    lines=['# Final results: load-transfer-aware palletizing','',
           '**Control targets pass at nominal 8 kg; full acceptance is not achieved.**','',
           'Official figures below come only from the unchanged evaluator through the integrity gate.',
           'All official 0/4/8 kg rows are in `FINAL_OFFICIAL.json` and evaluation text files.','',
           '## Official same-PC 8 kg integration','']+table(official)
    lines+=['','## Official collision delays (ms) at 8 kg','',
            '| Combination | Time s | 10 N | 15 N | 40 N | 100 N | Minimum | False max |',
            '|---|---:|---|---|---|---|---|---|']
    for name,combo in official.items():
        for c in combo['collisions']:
            if c['mass']==8:
                lines.append(f"| {name} | {c['time']} | {c['10N']} | {c['15N']} | {c['40N']} | {c['100N']} | {c['minimum']} | {c['false_max']} |")
    lines+=['','`놓침`=miss; `경보 중`=alarm already active; `구분 불가`=indistinguishable, not a detection.','',
            '## Historical official control ablation (8 kg)','',
            'These are preserved stage snapshots, not reruns of wrappers against the final shared class.',
            'All use the unchanged baseline detector. Sharing and support fusion are evaluated separately below.','',
            '| Stage | RMS/max mm | Pick/place N m | Table N | Saturation % |',
            '|---|---:|---:|---:|---:|']
    for label,filename in (
        ('A baseline','evaluation_baseline_servo.txt'), ('A C1','evaluation_D2_A_C1.txt'),
        ('A C1+ramp','evaluation_D2_A_ramp.txt'), ('A C1+C2','evaluation_D2_A_C1_C2.txt'),
        ('A C1+C2+C3','evaluation_D2_D3_servo_new_base.txt'),
        ('B baseline','evaluation_baseline_torque.txt'), ('B B1','evaluation_D2_B_B1.txt'),
        ('B B1+B2','evaluation_D2_B_B1_B2.txt'), ('B +ramp B3','evaluation_D2_B_ramp.txt'),
        ('B +B3','evaluation_D2_D3_torque_new_base.txt')):
        control,_=official_cells((R/filename).read_text());c=control[8.]
        lines.append(f'| {label} | {c[0]}/{c[1]} | {c[3]}/{c[4]} | {c[5]} | {c[2]} |')
    lines+=['','## Historical official detector ablation (baseline servo, 8 kg)','',
            '| Stage | False pick/place/other | Transport 10/15/40/100 N delay ms |',
            '|---|---|---|']
    for label,filename in (
        ('Baseline','evaluation_baseline_servo.txt'), ('+V','evaluation_D3_detector_v.txt'),
        ('+V+B','evaluation_D3_detector_vb.txt'), ('+V+B+W, instantaneous T','evaluation_D3_detector_vbw.txt'),
        ('D3 symmetric T','evaluation_D3_final_servo.txt'),
        ('Final FAST/mass gate','evaluation_final_servo_base_final.txt')):
        control,collisions=official_cells((R/filename).read_text());c=control[8.];delay=collisions[(8.,3.)]
        lines.append(f"| {label} | {'/'.join(c[6:9])} | {' / '.join(delay[:4])} |")
    lines+=['','Historical W and T stages use the documented D3 threshold snapshot; the final FAST protocol',
            'uses a new disjoint no-collision sample and mass-zero gate. This last row is a combined change,',
            'not a claim that persistence alone caused every difference. V alone reduced peak pickup Fz',
            '45.21 to 36.23 N, but oracle force RMSE worsened; no universal V-accuracy claim is made.','',
            '## Sharing and support fusion (supplemental)','',
            '| Mode | Support blend | Shared | RMS/max mm | Pick/place N m | Last20ms mean table N | Median/p99/max us |',
            '|---|---:|---|---:|---:|---:|---:|']
    for r in shared['rows']:
        t=r['timing']['combined']
        lines.append(f"| {r['mode']} | {r['blend']} | {r['shared']} | {r['rms_mm']:.3f}/{r['max_mm']:.3f} | "
                     f"{r['torque_step_pick_nm']:.2f}/{r['torque_step_place_nm']:.2f} | {r['table_last20ms_mean_n']:.2f} | "
                     f"{t['median_us']:.1f}/{t['p99_us']:.1f}/{t['max_us']:.1f} |")
    lines+=['','All four share-on/off parity comparisons are exact for physical arrays, residuals and alarms.',
            'Sharing advances the estimator 6000 instead of 12000 times; improvement in physical outputs',
            'comes from the estimator/load algorithm, not object reuse. Full support fusion was rejected:',
            'it increased servo place step and torque-controller tracking error; selected blend is 0.10.','',
            '## Robustness (supplemental, frozen sources; no retuning)','',
            '189 mass/COM references + 180 stratified force cases + 16 offset cases per architecture.',
            '7 masses including nonintegers, 27 COM positions, 3 phases with 5 times each, 6 force norms',
            '5–100 N, ±x/±y/±z and four diagonal directions. Not the full Cartesian product.','',
            '| Mode | No-collision FA episodes / 189 | Collision detected/missed/prior/indist. | Tool contacts missed / present | Worst max TCP mm | Worst p99/max us |',
            '|---|---:|---|---:|---:|---:|']
    for mode,data in robust.items():
        nc=data['no_collision'];counts=data['collision_counts'];t=data['timing']['combined']
        lines.append(f"| {mode} | {nc['with_false_alarm']}/189 | "+'/'.join(str(counts[k]) for k in ('detected','missed','alarm_before_collision','indistinguishable'))+
                     f" | {data['offset_contacts_missed']}/{data['offset_contact_episodes']} | {nc['worst_tcp_mm']:.2f} | {t['largest_p99_us']:.1f}/{t['observed_max_us']:.1f} |")
    failure_rows=[]
    for mode in ('servo','torque'):
        allrows=load('sweeps/d4-robust-'+mode+'/results.json')['results']
        for r in allrows:
            if r['kind']!='reference':continue
            reasons=[]
            if r['saturation_pct']>0:reasons.append('saturation')
            if r['diverged']:reasons.append('divergence')
            if r['false_alarms']>0:reasons.append('false_alarm')
            ratio=r['table_force_place_max_n']/(r['mass']*9.81) if r['mass'] else 0.
            if ratio>1.1:reasons.append('table_above_1.1mg')
            if reasons:
                failure_rows.append(dict(mode=mode,mass_kg=r['mass'],com_m=json.dumps(r['com']),
                    reasons=';'.join(reasons),saturation_pct=r['saturation_pct'],max_tcp_mm=r['max_mm'],
                    place_table_n=r['table_force_place_max_n'],table_over_mg=ratio,false_alarms=r['false_alarms']))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['mode','mass_kg','com_m','reasons',
        'saturation_pct','max_tcp_mm','place_table_n','table_over_mg','false_alarms'],lineterminator='\n')
    writer.writeheader();writer.writerows(failure_rows);(R/'D4_CONTROL_FAILURES.csv').write_text(stream.getvalue())
    lines+=['','Failure-map figures vary mass/COM/time across cells and must not be read as fixed-condition',
            'minimum detectable force curves. Default-noise finite coverage does not establish an event probability.',
            'Separate 50-case noise/friction/stiffness study: `D4_PERTURBATIONS.md`; its perturbed model',
            'figures are not official. Three noise seeds had no false alarms; stiffness/corner cases did.',
            'Low-mass force-ratio and any saturation failures are listed in `D4_CONTROL_FAILURES.csv`.',
            'All 12 genuine offset tool contacts across A/B were detected; other offset cases had no tool',
            'contact but some had false alarms. With unchanged baseline controllers, the historical 30 mm',
            'horizontal-offset misses became 12 ms (servo, 86.06 N contact) and 19 ms (torque, 29.66 N)',
            'with the final FAST detector; each also had one extra false rise.',
            'Torque friction/corner sensitivity reached 17.26 mm and 138.46 N; no perturbed case diverged or saturated.','',
            '## Acceptance','']
    for mode,g in goals.items():
        lines.append(f"### {mode}")
        lines.extend(f"- {'PASS' if v else 'MISS'}: {k}" for k,v in g['tests'].items())
        lines.append('- MISS: sustained pre-release handover; low-force sensitivity and some delay targets.')
        lines.append('- Overall full acceptance: **not achieved**.')
    lines+=['','## Attribution, rejected attempts and limits','',
            '- Historical C1/B1 alone improved tracking but worsened torque steps; continuous load FF greatly reduced switching-induced steps.',
            '- New velocity candidates failed a predeclared no-collision oracle-error guard; retained V is not universally more accurate.',
            '- FAST thresholds: 24 training + 12 disjoint holdout episodes, zero false alarms; no collision data used for tuning.',
            '- An earlier direction-confounded detector candidate was discarded and is not represented as a successful final result.',
            '- Early settle/overtravel/two-stage placement candidates worsened force/saturation/tracking and were rejected.',
            '- Offset-associated false alarms and support-line vertical ambiguities remain. No safety-stop response is implemented.',
            '- Finite empirical stability and desktop wall-clock timings are not passivity or hard-real-time proofs.',
            '- See `docs/METHOD.md`, `docs/LIMITATIONS.md`, `docs/REFERENCES.md` and presentation.pdf.','']
    (R/'FINAL_SUMMARY.md').write_text('\n'.join(lines))
    make_slides(official,robust,goals)
    print('Built final summary, ablations, goals, failure CSV, figures and 16-page presentation')


if __name__=='__main__':
    main()
