"""Synthetic selector checks for the exploratory variable-capacity control."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from revision2_reservoir_capacity import VariableReservoir
from revision2_selector import RANK_NAMESPACE, digest


def candidates(stage: str, seed: int) -> list[dict]:
    index = {"AF2": 1, "AF3": 2, "AF4": 3, "AF5": 4}[stage]
    return [{"example_id": f"{stage}-{i}", "task_id": stage,
             "stage_index": index, "label": i % 2,
             "coverage_key": f"{stage}-key-{i}",
             "random_rank_sha256": digest(seed, RANK_NAMESPACE, f"{stage}-{i}")}
            for i in range(400)]


class ReservoirCapacityTests(unittest.TestCase):
    def test_capacity_and_determinism(self):
        for capacity in (50, 200, 400):
            left = VariableReservoir("reservoir_capacity", 42, capacity)
            right = VariableReservoir("reservoir_capacity", 42, capacity)
            first = candidates("AF2", 42)
            second = candidates("AF3", 42)
            for buffer in (left, right):
                buffer.update(first, ("AF2",))
                self.assertEqual(buffer.seen_count, 400)
                self.assertEqual(len(buffer.entries), capacity)
                self.assertTrue(all(row["task_id"] == "AF2" for row in buffer.sample(4, 2)))
                buffer.update(second, ("AF2", "AF3"))
                self.assertEqual(buffer.seen_count, 800)
                self.assertEqual(len(buffer.entries), capacity)
                self.assertEqual(len({row["example_id"] for row in buffer.entries}), capacity)
            self.assertEqual(left.state_dict(), right.state_dict())
            resumed = VariableReservoir.from_state_dict(left.state_dict())
            self.assertEqual(resumed.state_dict(), left.state_dict())

    def test_future_and_duplicate_rejected(self):
        buffer = VariableReservoir("reservoir_capacity", 17, 50)
        first = candidates("AF2", 17)
        buffer.update(first, ("AF2",))
        with self.assertRaises(ValueError):
            buffer.update(first, ("AF2", "AF3"))
        with self.assertRaises(ValueError):
            buffer.sample(4, 1)


if __name__ == "__main__":
    unittest.main()
