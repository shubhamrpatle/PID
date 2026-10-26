"""Developer-only base-weight execution check, NOT final-checkpoint evaluation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pid_common import rows, atomic, digest, read, sha, require
from paired_eval import PairedBackend, pair_one
from models import Backend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-root', type=Path, help='Local organized_clean/hbb/detection/datasets directory')
    args = p.parse_args()
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    project = args.controls.resolve() / 'workspace/Zero_shot_ anlysis_LA'
    plan = dict(controls=str(args.controls.resolve()), budget=64,
                models={'base_la': dict(base=str(project / 'model/LocateAnything-3B/checkpoint'))})
    inventory = {e['path']: e['sha256'] for e in read(args.controls / 'transfer_inventory.json')['files']}
    examples = []
    for ds in ('DIOR', 'DOTAv2'):
        record = rows(project / 'runs/paper_rex_pointing_full_20260915_v1/assets' / (ds + '.jsonl'))[0]
        image_path = (args.image_root / ds.lower() / 'eval/images' / record['file_name']
                      if args.image_root else Path(record['image_path']))
        key = f"workspace/Zero_shot_ anlysis_LA/dataset/benchmark/{ds}/images/{record['file_name']}"
        require(sha(image_path) == inventory[key], 'Diagnostic image differs from frozen content')
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        examples.append((record, image))
    base = Backend('base_la', plan)
    base.name = 'pid'
    base.decoders = base.pid.make_decoders(base.model)
    model = PairedBackend(plan, backend=base)
    report = dict(purpose='Base-weight integration only, not final PID metrics or baseline timing',
                  device=torch.cuda.get_device_name(0), budget=64, pairs=[])
    for index, (record, image) in enumerate(examples):
        result = pair_one(model, record, image, index, digest(plan))
        report['pairs'].append(result)
        print(record['suite'], 'MATCHED DISCOVERY', len(result['modes']['packed']['points']),
              'extent forwards', {m: r['round2']['actual_round2_forwards'] for m, r in result['modes'].items()}, flush=True)
    # Exercise multi-address execution even if the unadapted model discovers few points.
    # These synthetic locations are an execution test, not benchmark predictions.
    from pixel_pivr.class_wave import ClassPointBranch
    from zero_shot_analysis.pixel_pivr_wave import AddressedCrop
    from evaluate_pixel_pivr_sequential_wave_balanced100 import compact_round2_prompt, point_to_norm
    point_decoder = model.routes['packed'][0]
    image = examples[1][1]
    prepared = point_decoder.prepare_image(image, [ClassPointBranch('ship', 'Point to all ships.', 123)])
    addresses = []
    for i, (x, y) in enumerate(((.2, .3), (.5, .5), (.8, .7))):
        point = [x * image.width, y * image.height]
        addresses.append(AddressedCrop(str(i), 'ship', None,
            compact_round2_prompt('ship', point, *image.size), point_normalized=point_to_norm(point, *image.size)))
    report['synthetic_address_kernel_check'] = {}
    for mode in ('packed', 'sequential'):
        answers, stats = model.routes[mode][1].decode_prepared_image(prepared, addresses, allow_none=True)
        require(len(answers) == 3, 'Synthetic execution dropped an address')
        require(stats['actual_round2_forwards'] == (1 if mode == 'packed' else 3), 'Wrong synthetic forward count')
        report['synthetic_address_kernel_check'][mode] = dict(answers=answers, stats=stats)
        print('SYNTHETIC KERNEL', mode, len(answers), 'addresses', stats['actual_round2_forwards'], 'forwards', flush=True)
    atomic(args.output, report)


if __name__ == '__main__':
    main()
