"""Selection must respect the worst condition, including checkpointed errors."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / 'skelkin').is_dir():
    from skelkin import condition_objects as conditions, main, project_handler
else:
    import condition_objects as conditions
    import main
    import project_handler


class MaximumSensitivityPolicyTests(unittest.TestCase):
    def errors(self, *rows):
        return conditions.ItemErrorList([
            conditions.ItemError(list(species), mean, maximum, 5.0)
            for species, mean, maximum in rows
        ])

    def test_low_mean_cannot_admit_a_group_above_maximum_threshold(self):
        errors = self.errors(
            (['A'], .003, .004),
            (['A', 'B'], .0044, .00696),
        )
        self.assertEqual(main._select_step2_omitted_species(errors, .005), ['A'])

    def test_largest_valid_group_wins_over_lower_error_smaller_group(self):
        errors = self.errors(
            (['A'], .0001, .0002),
            (['A', 'B', 'C'], .004, .0049),
            (['A', 'B'], .002, .003),
        )
        self.assertEqual(main._select_step2_omitted_species(errors, .005), ['A', 'B', 'C'])

    def test_maximum_equal_to_threshold_is_accepted(self):
        errors = self.errors((['A', 'B'], .002, .005))
        self.assertEqual(main._select_step2_omitted_species(errors, .005), ['A', 'B'])

    def test_no_group_under_maximum_threshold_means_no_removal(self):
        errors = self.errors((['A'], .001, .006))
        self.assertEqual(main._select_step2_omitted_species(errors, .005), [])

    def test_nonfinite_maxima_cannot_pass_the_threshold(self):
        errors = self.errors(
            (['A'], .001, float('nan')),
            (['A', 'B'], .001, float('inf')),
            (['A', 'B', 'C'], .001, float('-inf')),
        )
        self.assertEqual(main._select_step2_omitted_species(errors, .005), [])

    def test_checkpoint_second_error_column_controls_selection(self):
        old_identifier = project_handler.project_identifier
        try:
            with tempfile.TemporaryDirectory() as folder:
                project_handler.project_identifier = str(Path(folder) / 'policy')
                Path(project_handler.project_identifier + '.skn').write_text('', encoding='utf-8')
                project_handler.write_step_error(2, self.errors(
                    (['A'], .003, .005),
                    (['A', 'B'], .0044, .00696),
                ))
                loaded = project_handler.get_me_data('step 2')
                self.assertEqual(len(loaded.items), 2)
                self.assertEqual(loaded.items[1].get_max_value(), .00696)
                self.assertEqual(main._select_step2_omitted_species(loaded, .005), ['A'])
        finally:
            project_handler.project_identifier = old_identifier


if __name__ == '__main__':
    unittest.main()
