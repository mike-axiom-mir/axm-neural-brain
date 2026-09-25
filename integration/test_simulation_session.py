from copy import deepcopy
import json
import unittest

from neural.axm_brain import AXMBrain, BrainConfig
from neural.axm_brain.simulation_session import SimulationSession
from neural.axm_brain.state import snapshot_payload
from axm_neural_network.microsim import MicroDynamics, DynamicsRules


class SimulationSessionTests(unittest.TestCase):
    def session(self, policy='fixed', providers=None):
        brain = AXMBrain(BrainConfig(3,8,2,seed=41,learning_rate=.04,replay_capacity=16))
        providers = providers or {'slow':MicroDynamics(DynamicsRules(horizon=3)),
                                 'fast':MicroDynamics(DynamicsRules(action_gain=.9,horizon=3))}
        return SimulationSession(brain,providers,range(8),range(100,108),policy=policy,max_transitions=300)

    def test_json_resume_matches_uninterrupted_for_every_policy(self):
        for policy in SimulationSession.POLICIES:
            with self.subTest(policy=policy):
                full,split = self.session(policy),self.session(policy)
                full.advance(16)
                split.advance(7)
                exported = json.loads(json.dumps(split.to_snapshot(),sort_keys=True))
                split = SimulationSession.from_snapshot(exported,split.providers)
                split.advance(9)
                self.assertEqual(full.to_snapshot(),split.to_snapshot())
                before = split.to_snapshot()
                self.assertEqual(full.evaluate(),split.evaluate())
                self.assertEqual(before,split.to_snapshot())
                self.assertTrue(split.reproduce())

    def test_held_out_changes_never_change_training_choices_or_weights(self):
        a,b = self.session('error_guided'),self.session('error_guided')
        b.evaluation_seeds = tuple(range(900,908))
        a.evaluate()
        a.advance(15)
        b.advance(15)
        self.assertEqual(a.records,b.records)
        self.assertEqual(a.learner.to_snapshot(),b.learner.to_snapshot())

    def test_bad_late_transition_rolls_back_the_complete_call(self):
        class Broken(MicroDynamics):
            calls = 0
            def step(self,state,action):
                self.calls += 1
                result = super().step(state,action)
                if self.calls==8: result['experience']['body']['target'][0] += 1
                return result
        session = self.session(providers={'broken':Broken(DynamicsRules(horizon=3))})
        before = session.to_snapshot()
        with self.assertRaises(ValueError): session.advance(5)
        self.assertEqual(before,session.to_snapshot())

    def test_rehashed_scheduler_accounting_and_provider_changes_rejected(self):
        session = self.session('random')
        session.advance(8)
        body = session.to_snapshot()['body']
        for field,value in [('rng_state',2),('transitions',1),('automatic_adoption',True),
                            ('family_order',['fast','slow']),('counts',{'slow':-1,'fast':9})]:
            altered = deepcopy(body)
            altered[field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                SimulationSession.from_snapshot(snapshot_payload(altered),session.providers)
        changed = dict(session.providers)
        changed['slow'] = MicroDynamics(DynamicsRules(horizon=2))
        with self.assertRaises(ValueError): SimulationSession.from_snapshot(session.to_snapshot(),changed)

    def test_budget_overlap_and_dimensions_fail_before_learning(self):
        session = self.session()
        before = session.to_snapshot()
        with self.assertRaises(ValueError): session.advance(101)
        self.assertEqual(before,session.to_snapshot())
        with self.assertRaises(ValueError):
            SimulationSession(session.learner,session.providers,[1,2],[2,3])
        with self.assertRaises(ValueError):
            SimulationSession(AXMBrain(BrainConfig(2,3,1)),session.providers,[1],[2])
