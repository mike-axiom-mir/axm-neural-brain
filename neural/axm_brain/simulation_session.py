"""Resumable learning from host-selected, replayable simulation providers.

Import this integration module only when axm-neural-network is available.
The host owns persistence and invocation. The continuing parent is never changed.
"""
from copy import deepcopy
import math

from axm_neural_network.simulation_contract import describe, verified_transition
from .core import AXMBrain, Experience, XorShift64
from .simulation import _reset, _seeds
from .state import snapshot_payload, verify_snapshot


def _integer(value, name, low=0, high=100_000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


class SimulationSession:
    """A persistent descendant plus a fixed, random or error-guided curriculum.

    error_guided is an explicit scheduling heuristic: after one episode per
    family, sample in proportion to moving pre-update prediction error plus .01.
    It is not a trained meta-network, and improvement over controls is empirical.
    Held-out results never enter scheduling. All providers must be pure/replayable.
    """
    SCHEMA = 'axm.simulation-session/v1'
    POLICIES = ('fixed', 'random', 'error_guided')

    def __init__(self, parent, providers, training_seeds, evaluation_seeds, *,
                 policy='fixed', scheduler_seed=1, max_transitions=10_000):
        if not isinstance(providers, dict) or not 1 <= len(providers) <= 16:
            raise ValueError('one to sixteen named providers required')
        if any(not isinstance(k,str) or not k or len(k)>128 for k in providers):
            raise ValueError('provider names must be nonempty bounded strings')
        if policy not in self.POLICIES:
            raise ValueError('unknown curriculum policy')
        _integer(scheduler_seed, 'scheduler seed', 0, XorShift64.MASK)
        _integer(max_transitions, 'transition budget', 1)
        self.providers = dict(providers)
        self.specs = {key:describe(value) for key,value in providers.items()}
        for spec in self.specs.values():
            if (spec['observation_size'],spec['target_size']) != (parent.config.input_size,parent.config.output_size):
                raise ValueError('provider dimensions do not match learner')
        self.training_seeds = _seeds(training_seeds,'training seeds')
        self.evaluation_seeds = _seeds(evaluation_seeds,'evaluation seeds')
        if len(self.training_seeds)>10_000 or len(self.evaluation_seeds)>10_000:
            raise ValueError('seed list exceeds session bound')
        if set(self.training_seeds) & set(self.evaluation_seeds):
            raise ValueError('training and held-out seeds overlap')
        self.parent = parent.to_snapshot()
        self.learner = AXMBrain.from_snapshot(self.parent)
        self.learner.wake()
        self.policy, self.scheduler_seed, self.max_transitions = policy,scheduler_seed,max_transitions
        self.rng = XorShift64(scheduler_seed)
        self.counts = {key:0 for key in self.specs}
        self.errors = {key:0.0 for key in self.specs}
        self.records = []
        self.transitions = 0

    def _choose(self):
        names = list(self.specs)
        if self.policy == 'fixed':
            return names[len(self.records)%len(names)]
        if self.policy == 'random':
            return names[min(int(self.rng.uniform(0,len(names))),len(names)-1)]
        for name in names:
            if self.counts[name] == 0:
                return name
        weights = [self.errors[key]+.01 for key in names]
        point = self.rng.uniform(0,sum(weights))
        for name,weight in zip(names,weights):
            point -= weight
            if point <= 0: return name
        return names[-1]

    def _episode(self, name, seed, occurrence, *, train):
        provider, spec = self.providers[name], self.specs[name]
        # Family-local occurrence gives comparable actions regardless of when a
        # curriculum visits the family. Policy RNG is independent of action RNG.
        actions = XorShift64((seed ^ (occurrence*104729) ^ 0xA5C3) & XorShift64.MASK)
        brain = self.learner if train else AXMBrain.from_snapshot(self.learner.to_snapshot())
        state = provider.reset(seed)
        packets, loss, chosen_actions = [], 0.0, []
        for index in range(spec['horizon']):
            action = actions.uniform(*spec['action_bounds'])
            result = provider.step(state,action)
            event = verified_transition(provider,result['experience'],spec=spec)
            if event['before'] != provider.snapshot(state)['body']['state'] or event['action'] != action:
                raise ValueError('provider substituted the requested state or action')
            if event['seed'] != seed or event['terminal'] != (index+1==spec['horizon']):
                raise ValueError('provider seed or episode boundary mismatch')
            # Ensure the returned state really is the packet's next state.
            state = provider.restore(provider.snapshot(result['state']))
            if provider.snapshot(state)['body']['state'] != event['after']:
                raise ValueError('provider returned state differs from verified outcome')
            _reset(brain)
            prediction = brain.predict(event['observation'],update_state=False)
            loss += sum((a-b)**2 for a,b in zip(prediction,event['target']))/spec['target_size']
            if train:
                brain.experience(Experience(event['observation'],target=event['target'],
                                            source=event['experience_source'],tag=result['experience']['sha256']))
            packets.append(result['experience']['sha256'])
            chosen_actions.append(action)
        return loss/spec['horizon'], snapshot_payload({'packets':packets})['sha256'], chosen_actions

    def advance(self, episodes):
        """Commit the whole bounded call only after every transition verifies."""
        _integer(episodes,'episodes',1,10_000)
        if len(self.records)+episodes > 10_000:
            raise ValueError('episode history bound exceeded')
        # Reserve the worst case before invoking a provider or changing state.
        if self.transitions+episodes*max(s['horizon'] for s in self.specs.values()) > self.max_transitions:
            raise ValueError('call exceeds remaining transition budget')
        work = self.from_snapshot(self.to_snapshot(),self.providers)
        start = len(work.records)
        for _ in range(episodes):
            name = work._choose()
            occurrence = work.counts[name]
            seed = work.training_seeds[occurrence%len(work.training_seeds)]
            before = work.learner.to_snapshot()['sha256']
            error, trace, actions = work._episode(name,seed,occurrence,train=True)
            if not math.isfinite(error): raise ValueError('nonfinite learning error')
            work.errors[name] = error if occurrence==0 else .8*work.errors[name]+.2*error
            work.counts[name] += 1
            work.transitions += work.specs[name]['horizon']
            record = snapshot_payload({
                'episode':len(work.records), 'family':name, 'seed':seed,
                'occurrence':occurrence, 'actions':actions, 'pre_update_mse':error,
                'transition_count':work.specs[name]['horizon'], 'trace_sha256':trace,
                'learner_before':before,'learner_after':work.learner.to_snapshot()['sha256'],
                'previous':work.records[-1]['sha256'] if work.records else None,
            })
            work.records.append(record)
        self.__dict__.update(work.__dict__)
        return deepcopy(self.records[start:])

    def evaluate(self):
        """Read-only held-out evaluation; costs are separate from training."""
        result = {}
        for name,spec in self.specs.items():
            errors = [self._episode(name,seed,0,train=False)[0] for seed in self.evaluation_seeds]
            result[name] = {'mse':sum(errors)/len(errors),
                            'transitions':len(errors)*spec['horizon']}
        return {'families':result,'mean_family_mse':sum(r['mse'] for r in result.values())/len(result),
                'transitions':sum(r['transitions'] for r in result.values()),
                'experience_source':'deterministic_simulation'}

    def to_snapshot(self):
        return snapshot_payload({
            'schema':self.SCHEMA, 'parent':self.parent, 'learner':self.learner.to_snapshot(),
            'providers':self.specs, 'family_order':list(self.specs),
            'training_seeds':list(self.training_seeds),'evaluation_seeds':list(self.evaluation_seeds),
            'policy':self.policy,'scheduler_seed':self.scheduler_seed,'rng_state':self.rng.state,
            'max_transitions':self.max_transitions,'transitions':self.transitions,
            'counts':self.counts,'errors':self.errors,'records':self.records,
            'adoption_state':'experimental-descendant','automatic_adoption':False,
        })

    @classmethod
    def from_snapshot(cls, snapshot, providers):
        body = verify_snapshot(snapshot)
        if body.get('schema') != cls.SCHEMA or body.get('automatic_adoption') is not False or body.get('adoption_state')!='experimental-descendant':
            raise ValueError('session schema or adoption contract mismatch')
        session = cls(AXMBrain.from_snapshot(body['parent']),providers,body['training_seeds'],
                      body['evaluation_seeds'],policy=body['policy'],scheduler_seed=body['scheduler_seed'],
                      max_transitions=body['max_transitions'])
        if session.specs != body['providers'] or list(session.specs)!=body['family_order']:
            raise ValueError('session provider identity, order or parameters changed')
        session.learner = AXMBrain.from_snapshot(body['learner'])
        if session.learner.config != AXMBrain.from_snapshot(session.parent).config:
            raise ValueError('learner configuration differs from parent')
        _integer(body['rng_state'],'RNG state',1,XorShift64.MASK)
        records = body['records']
        if not isinstance(records,list) or len(records)>10_000:
            raise ValueError('invalid episode history')
        counts, errors, transitions, previous = dict(session.counts),dict(session.errors),0,None
        initial = AXMBrain.from_snapshot(session.parent)
        initial.wake()
        expected_before = initial.to_snapshot()['sha256']
        for index,record in enumerate(records):
            event = verify_snapshot(record)
            name = event['family']
            if name not in counts or event['episode']!=index or type(event['episode']) is not int or event['previous']!=previous:
                raise ValueError('invalid episode chain')
            if name != session._choose():
                raise ValueError('recorded family differs from curriculum decision')
            if type(event['occurrence']) is not int or event['occurrence']!=counts[name] or type(event['seed']) is not int or event['seed']!=session.training_seeds[counts[name]%len(session.training_seeds)]:
                raise ValueError('invalid episode seed or occurrence')
            horizon = session.specs[name]['horizon']
            if type(event['transition_count']) is not int or event['transition_count']!=horizon or len(event['actions'])!=horizon or event['learner_before']!=expected_before:
                raise ValueError('invalid episode transition accounting')
            actions = XorShift64((event['seed'] ^ (counts[name]*104729) ^ 0xA5C3) & XorShift64.MASK)
            if event['actions'] != [actions.uniform(*session.specs[name]['action_bounds']) for _ in range(horizon)]:
                raise ValueError('recorded actions differ from the declared curriculum')
            value = event['pre_update_mse']
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('invalid episode learning error')
            errors[name] = value if counts[name]==0 else .8*errors[name]+.2*value
            counts[name] += 1
            session.counts, session.errors = dict(counts), dict(errors)
            session.records.append(record)
            transitions += horizon
            previous,expected_before = record['sha256'],event['learner_after']
        if session.rng.state != body['rng_state'] or transitions != body['transitions'] or type(body['transitions']) is not int or transitions>session.max_transitions or counts!=body['counts'] or errors!=body['errors']:
            raise ValueError('session accounting differs from episode history')
        if expected_before != session.learner.to_snapshot()['sha256']:
            raise ValueError('session learner is not the recorded descendant')
        parent_count = body['parent']['body']['state']['host_experience_count']
        if session.learner.host_experience_count != parent_count+transitions:
            raise ValueError('learner update count differs from simulated experience')
        session.counts,session.errors,session.transitions,session.records = counts,errors,transitions,deepcopy(records)
        # Exact schema comparison also rejects omitted/extra fields and coercions.
        if session.to_snapshot() != snapshot:
            raise ValueError('noncanonical or inconsistent session snapshot')
        return session

    def reproduce(self):
        """Replay all learning to verify causal evidence, beyond stored hashes."""
        rebuilt = type(self)(AXMBrain.from_snapshot(self.parent),self.providers,self.training_seeds,
                             self.evaluation_seeds,policy=self.policy,scheduler_seed=self.scheduler_seed,
                             max_transitions=self.max_transitions)
        # Calls originally could have used different sizes; the controller's
        # sequence is independent of those persistence boundaries.
        if len(self.records)*max(s['horizon'] for s in self.specs.values())<=self.max_transitions:
            if self.records: rebuilt.advance(len(self.records))
        else:
            for _ in self.records: rebuilt.advance(1)
        return rebuilt.to_snapshot() == self.to_snapshot()
