from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_saved import image_dimensions
from native_parser import la_native, reparse
from scoring import score


class SavedReportTests(unittest.TestCase):
    def test_clamped_degenerate_box_replays_exactly(self):
        specs = [dict(canonical='small vehicle', prompt='small vehicle')]
        answer = '<box><165><1000><170><1000></box>'
        predictions, _ = la_native(answer, specs, (1024, 1024), 'boxes', 'small vehicle')
        record = dict(uid='r', width=1024.0, height=1024.0,
                      gt=[dict(label='small vehicle', hbox=[160, 900, 180, 950])])
        before = deepcopy(record)
        result = dict(uid='r', query=dict(class_specs=specs), predictions=predictions,
                      refinements=[dict(label='small vehicle', answer=answer)])
        legacy = reparse('pid', record, result, 'boxes', True)
        self.assertNotEqual(legacy['predictions'], predictions)
        replay = reparse('pid', image_dimensions(record), result, 'boxes', True)
        self.assertEqual(replay['predictions'], predictions)
        self.assertEqual(record, before)
        metrics, per = score([record], [result], 'boxes')
        self.assertEqual(metrics['invalid_geometry_outputs'], 1)
        self.assertEqual(per[0]['scores']['0.5']['class_aware']['fp'], 1)
        other, _ = score([record], [legacy], 'boxes')
        self.assertEqual(metrics, other)

    def test_do_not_truncate_invalid_image_dimensions(self):
        for width in (1023.5, 0, -1, float('nan'), float('inf'), '1024', True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                image_dimensions(dict(width=width, height=1024))

    def test_valid_boxes_unchanged(self):
        specs = [dict(canonical='ship', prompt='ship')]
        record = dict(width=1024., height=1024.)
        result = dict(query=dict(class_specs=specs), refinements=[
            dict(label='ship', answer='<box><100><200><300><400></box>')])
        self.assertEqual(reparse('pid', record, result, 'boxes', True),
                         reparse('pid', image_dimensions(record), result, 'boxes', True))


if __name__ == '__main__':
    unittest.main()
