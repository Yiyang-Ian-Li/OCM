import copy
import re
import unittest
from unittest.mock import Mock

from benchmarks.sciworld.agentboard import CorrectedAgentBoard, PROTOCOL, correct_label, read_event
from benchmarks.sciworld.env import SciWorldEnv
from benchmarks.sciworld.task import SciWorldTask


def event(kind, **objects):
    return {"kind": kind, "bindings": {key: {"name": name, "id": uid}
                                       for key, (name, uid) in objects.items()}}


class AgentBoardCorrectionTest(unittest.TestCase):
    def test_sampling_and_unaffected_labels_are_preserved(self):
        original, _ = SciWorldTask.load_tasks("agentboard_test")
        corrected, count = SciWorldTask.load_tasks("agentboard_corrected")
        original, corrected = list(original), list(corrected)
        self.assertEqual(count, 90)
        changed = []
        for index, (old, new) in enumerate(zip(original, corrected)):
            self.assertEqual((old.sub_task_name, old.variation_idx), (new.sub_task_name, new.variation_idx))
            self.assertEqual(len(old.subgoals), len(new.subgoals))
            self.assertEqual(new.evaluation_protocol, PROTOCOL)
            self.assertNotEqual(old.task_id, new.task_id)
            if old.subgoals != new.subgoals or old.goal_override != new.goal_override:
                changed.append(index)
        self.assertEqual(changed, [11, 17, 51, 52, 76])
        self.assertIn("below 10.0", corrected[11].goal_override)
        self.assertIn("below 50.0", original[11].goal_override)
        self.assertTrue(re.search(corrected[76].subgoals[1], "You focus on the book (Sherlock Holmes)."))
        self.assertFalse(re.search(corrected[76].subgoals[1], "You focus on the book (Pride and Prejudice)."))
        self.assertTrue(re.search(corrected[17].subgoals[3], "You move to the living room."))

    def lifespan(self, both=False):
        name = "lifespan-longest-lived-then-shortest-lived" if both else "lifespan-shortest-lived"
        patterns = ["You move to the outside."]
        if both:
            patterns.append("You focus on the baby brown bear.")
        patterns.append("You focus on the baby mouse.")
        return CorrectedAgentBoard(name, patterns, {"long": "brown bear", "short": "mouse"},
                                   {"brown bear": {1}, "mouse": {2}})

    def test_lifespan_accepts_stages_but_not_wrong_animals_or_failed_focus(self):
        for stage in ("baby", "juvenile", "adult"):
            scorer = self.lifespan()
            scorer.update(f"You focus on the {stage} mouse.", event("ActionFocus", obj=("mouse", 2)))
            self.assertEqual(scorer.progress, [False, True])
        scorer = self.lifespan()
        scorer.update("You focus on the adult brown bear.", event("ActionFocus", obj=("brown bear", 1)))
        scorer.update("No known action matches that input.", event("ActionFocus", obj=("mouse", 2)))
        scorer.update("You focus on the baby mouse.", None)
        self.assertEqual(scorer.progress, [False, False])

    def test_lifespan_preserves_explicit_longest_then_shortest_order(self):
        scorer = self.lifespan(both=True)
        scorer.update("You move to the outside.")
        scorer.update("You focus on the baby mouse.", event("ActionFocus", obj=("mouse", 2)))
        self.assertEqual(scorer.progress, [True, False, False])
        scorer.update("You focus on the adult brown bear.", event("ActionFocus", obj=("brown bear", 1)))
        self.assertEqual(scorer.progress, [True, True, False])
        scorer.update("You focus on the adult mouse.", event("ActionFocus", obj=("mouse", 2)))
        self.assertTrue(all(scorer.progress))

    def thermometer(self):
        patterns = ["You move to the living room.",
                    r"the thermometer measures a temperature of (-?\d+) degrees celsius",
                    "You move the unknown substance B to the purple box"]
        return CorrectedAgentBoard("use-thermometer", patterns,
                                   {"objectName": "unknown substance B", "temperaturePoint": 50,
                                    "boxAbove": "yellow box", "boxBelow": "purple box"},
                                   {"unknown substance B": {1}})

    def measure(self, scorer, temperature, uid=1):
        scorer.update(f"the thermometer measures a temperature of {temperature} degrees celsius",
                      event("ActionUseDevice", device=("thermometer", 3), patient=("unknown substance B", uid)))

    def place(self, scorer, box, uid=1):
        scorer.update(f"You move the unknown substance B to the {box}.",
                      event("ActionMoveObject", obj=("unknown substance B", uid), moveTo=(box, 4)))

    def test_temperature_uses_measured_branch_and_requires_target_measurement(self):
        for temperature, correct, wrong in [(66, "yellow box", "purple box"), (46, "purple box", "yellow box")]:
            scorer = self.thermometer()
            self.place(scorer, correct)
            self.assertFalse(any(scorer.progress))
            self.measure(scorer, temperature, uid=9)
            self.place(scorer, correct, uid=9)
            self.assertFalse(any(scorer.progress))
            self.measure(scorer, temperature)
            self.place(scorer, wrong)
            self.assertEqual(scorer.progress, [False, True, False])
            self.place(scorer, correct)
            self.assertEqual(scorer.progress, [False, True, True])
            scorer.update("You move to the living room.")
            self.assertTrue(all(scorer.progress))

    def test_latest_reading_is_used_and_failed_moves_do_not_count(self):
        scorer = self.thermometer()
        self.measure(scorer, 66)
        self.measure(scorer, 46)
        self.place(scorer, "yellow box")
        self.assertFalse(scorer.progress[-1])
        scorer.update("That can't be moved there.", event("ActionMoveObject", obj=("unknown substance B", 1),
                                                         moveTo=("purple box", 4)))
        self.assertFalse(scorer.progress[-1])
        self.place(scorer, "purple box")
        self.assertTrue(scorer.progress[-1])

    def test_equal_temperature_accepts_either_boundary_like_native(self):
        for box in ("yellow box", "purple box"):
            scorer = self.thermometer()
            self.measure(scorer, 50)
            self.place(scorer, box)
            self.assertTrue(scorer.progress[-1])

    def test_label_fix_does_not_mutate_release_input(self):
        row = {"id": 17, "goal": "goal", "subgoals": [",You move to the living room"]}
        original = copy.deepcopy(row)
        correct_label(row)
        self.assertEqual(row, original)

    def test_task_command_uses_the_same_goal_as_reset_only_for_corrected_protocol(self):
        for protocol, expected in [(PROTOCOL, "Rewritten goal."), ("agentboard", "Native goal.")]:
            fake = Mock()
            fake.reset.return_value = ("A room.", {"taskDesc": "Native goal."})
            fake.step.return_value = ("Task description:\nNative goal.", 0, False, {})
            task = SciWorldTask(sub_task_name="boil", variation_idx=5,
                                goal_override="Rewritten goal.", subgoals=["unmatched"],
                                evaluation_protocol=protocol)
            environment = SciWorldEnv(task, env=fake)
            environment.reset()
            self.assertEqual(environment.step("task").observation, "Task description:\n" + expected)

    def test_automatic_info_actions_cannot_reuse_a_previous_focus(self):
        automatic = Mock()
        automatic.getClass.return_value.getSimpleName.return_value = "ActionTaskDesc"
        agent = Mock()
        agent.getActionHistorySince.return_value = [automatic]
        self.assertIsNone(read_event(agent, 20))
        agent.getActionHistorySince.assert_called_once_with(20)
