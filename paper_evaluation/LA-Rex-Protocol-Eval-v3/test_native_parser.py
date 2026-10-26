"""Native LA postprocessing tests; no model, GPU, or inference required."""
import unittest
from native_parser import la_native, released_parser, reparse
from scoring import score

SPEC = [dict(canonical='ship', prompt='ship')]


class NativeParserTests(unittest.TestCase):
    def test_source_corner_repair(self):
        raw = '<ref>ship</ref><box><800><900><200><100></box>'
        expected = released_parser()(raw, 100, 200)['ship'][0]
        self.assertEqual(expected, [20, 20, 80, 180])
        self.assertEqual(la_native(raw, SPEC, (100, 200), 'boxes')[0][0]['hbox'], expected)

    def test_native_boundary_and_numeric_range(self):
        p, _ = la_native('<ref>ship</ref><box><0><0><1200><1500></box>', SPEC, (100, 200), 'boxes')
        self.assertEqual(p[0]['hbox'], [0, 0, 99, 199])
        p, _ = la_native('<ref>ship</ref><box><1000><1000></box>', SPEC, (100, 200), 'pointing')
        self.assertEqual(p[0]['point'], [99, 199])

    def test_native_fractional_tokens(self):
        p, _ = la_native('<ref>ship</ref><box><100.5><200.5></box>', SPEC, (1000, 1000), 'pointing')
        self.assertEqual(p[0]['point'], [100.5, 200.5])

    def test_no_missing_label_repair_for_direct_la(self):
        self.assertEqual(la_native('<box><0><0><500><500></box>', SPEC, (100, 100), 'boxes')[0], [])

    def test_address_identity_without_gt(self):
        p, _ = la_native('<box><0><0><500><500></box>', SPEC, (100, 100), 'boxes', 'ship')
        self.assertEqual(p, [dict(label='ship', hbox=[0, 0, 50, 50])])

    def test_wrong_explicit_label_not_overwritten_by_address(self):
        p, _ = la_native('<ref>car</ref><box><0><0><500><500></box>', SPEC, (100, 100), 'boxes', 'ship')
        self.assertEqual(p[0]['label'], '__unmapped__')

    def test_collapsed_or_wrong_task_output_is_fp(self):
        for raw in ('<box><0><0><0><0></box>', '<box><100><100></box>'):
            p, _ = la_native('<ref>ship</ref>' + raw, SPEC, (100, 100), 'boxes')
            self.assertTrue(p[0]['invalid_geometry'])

    def test_native_and_strict_use_identical_raw_answers(self):
        record = dict(uid='1', width=100, height=100, gt=[dict(label='ship', hbox=[10, 10, 80, 80])])
        result = dict(uid='1', query=dict(class_specs=SPEC), answers=[dict(queried_categories=['ship'],
            raw_response='<ref>ship</ref><box><800><800><100><100></box>')])
        native = reparse('standard', record, result, 'boxes', True)
        strict = reparse('standard', record, result, 'boxes', False)
        self.assertIs(native['answers'], strict['answers'])
        for output, tp in ((native, 1), (strict, 0)):
            scored, _ = score([record], [output], 'boxes')
            self.assertEqual(scored['scores']['class_aware']['0.5']['micro']['tp'], tp)

    def test_query_group_limits_labels(self):
        result = dict(query=dict(class_specs=SPEC + [dict(canonical='car', prompt='car')]),
                      answers=[dict(queried_categories=['ship'], raw_response='<ref>car</ref><box><100><100></box>')])
        p = reparse('standard', dict(width=100, height=100), result, 'pointing', True)
        self.assertEqual(p['predictions'][0]['label'], '__unmapped__')

    def test_other_models_keep_their_native_interface(self):
        result = {'predictions': []}
        self.assertIs(reparse('rex_omni', {}, result, 'boxes', True), result)

    def test_isolated_direct_point_has_query_identity(self):
        result = dict(query=dict(class_specs=SPEC), answers=[dict(queried_categories=['ship'],
                      raw_response='<box><100><100></box>')])
        parsed = reparse('class_isolated', dict(width=100, height=100), result, 'pointing', True)
        self.assertEqual(parsed['predictions'], [dict(label='ship', point=[10, 10])])


if __name__ == '__main__':
    unittest.main()
