#!/usr/bin/env python3
"""Print completed scores only; never mixes old and new evaluation protocols."""
import argparse
from pathlib import Path
from common import HERE,read


def render(summary):
    lines=['# GT-Present-Category RS Evaluation','',
        'Category-conditioned diagnostic; not a full-vocabulary detection benchmark.',
        'LA-family outputs use the pinned LA-native parser. Strict-parser sensitivity is saved separately.',
        'Percent scores. mF1 averages ten IoU thresholds, 0.50:0.05:0.95.',
        'Time is synchronized seconds per complete image, including every requested category.',
        'Valid outputs/s counts usable geometry; correct/s counts matches at IoU 0.50 or point containment.',
        'Invalid predictions remaining after native parsing count as false positives; attempted-output counts remain in summary.json.','']
    for kind in ('boxes','pointing'):
        for dataset in ('DIOR','DOTAv2'):
            lines+=['## '+dataset+' '+kind,'']
            if kind=='boxes':
                lines+=['| Model | F1@.50 | F1@.75 | mF1 | s/image | valid outputs/s | correct/s |',
                        '|---|---:|---:|---:|---:|---:|---:|']
            else:
                lines+=['| Model | Precision | Recall | F1@Point | s/image | valid outputs/s | correct/s |',
                        '|---|---:|---:|---:|---:|---:|---:|']
            for key,r in summary.items():
                if r['task']!=kind or r['dataset']!=dataset: continue
                if not r['complete']: raise ValueError('Incomplete row')
                s=r['scores']['class_aware' if kind=='boxes' else 'released']
                scores=[s['0.5']['f1'],s['0.75']['f1'],s['mean_f1_50_95']] if kind=='boxes' else [s['point'][k] for k in ('precision','recall','f1')]
                cells=[r['model']]+[f'{100*v:.2f}' for v in scores]+[f'{r[k]:.3f}' for k in ('seconds_per_complete_image','valid_geometry_outputs_per_second','correct_outputs_per_second')]
                lines.append('| '+' | '.join(cells)+' |')
            lines.append('')
    lines+=['Primary box scores above use class-aware matching. The released class-flattened',
            'compatibility scores, strict_sensitivity and detector sensitivity variants remain in summary.json.',
            'Smoke summaries are functional checks, not benchmark results.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    print(render(read(a.output/('smoke_summary.json' if a.smoke else 'summary.json'))))
