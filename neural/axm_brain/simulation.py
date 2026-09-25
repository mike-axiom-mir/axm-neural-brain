"""Direct, explicit micro-simulation experience for a cloned AXM brain.

The host supplies the simulator protocol. No simulator, network, file, process or
training activity is started on import. Source labels are preserved, not rewards.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import random
import time

from .core import AXMBrain, Experience
from .state import snapshot_payload


def _seeds(values, name):
    values = tuple(values)
    if not values or any(type(v) is not int for v in values) or len(set(values)) != len(values):
        raise ValueError(f'{name} must contain distinct integer seeds')
    return values


def _reset(brain):
    brain.hidden = [0.0]*brain.config.hidden_size
    brain.last_output = [0.0]*brain.config.output_size


def _batches(simulator, seeds, batch_size, epoch):
    for start in range(0,len(seeds),batch_size):
        batch = seeds[start:start+batch_size]
        states = [simulator.reset(seed) for seed in batch]
        randoms = [random.Random(seed ^ (epoch*104729) ^ 0xA5C3) for seed in batch]
        for _ in range(simulator.rules.horizon):
            actions = [rng.uniform(-1,1) for rng in randoms]
            results = simulator.step_many(states,actions)
            states = [result['state'] for result in results]
            yield results


def simulation_error(brain, simulator, seeds, *, batch_size=8):
    if (brain.config.input_size,brain.config.output_size) != (3,2):
        raise ValueError('reference micro-dynamics requires three observations and two predictions')
    seeds = _seeds(seeds,'evaluation seeds')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch size must be a positive integer')
    probe = AXMBrain.from_snapshot(brain.to_snapshot())
    total, count = 0.0, 0
    for batch in _batches(simulator,seeds,batch_size,0):
        for item in batch:
            event = simulator.verify_transition(item['experience'])
            _reset(probe)
            prediction = probe.predict(event['observation'],update_state=False)
            total += sum((a-b)**2 for a,b in zip(prediction,event['target']))/len(prediction)
            count += 1
    return {'mse':total/count,'transitions':count,'experience_source':'deterministic_simulation'}


def learn_in_simulation(parent, simulator, training_seeds, evaluation_seeds, *,
                        epochs=4, batch_size=8, max_transitions=10_000):
    """Each simulated batch teaches immediately, preserving the continuing parent."""
    training_seeds, evaluation_seeds = _seeds(training_seeds,'training seeds'), _seeds(evaluation_seeds,'evaluation seeds')
    if set(training_seeds) & set(evaluation_seeds):
        raise ValueError('training and evaluation seeds overlap')
    for name,value in (('epochs',epochs),('batch_size',batch_size),('max_transitions',max_transitions)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if (parent.config.input_size,parent.config.output_size) != (3,2):
        raise ValueError('reference micro-dynamics requires three observations and two predictions')
    training_count = len(training_seeds)*simulator.rules.horizon*epochs
    if training_count > max_transitions:
        raise ValueError('training exceeds the declared transition budget')
    original = parent.to_snapshot()
    candidate = AXMBrain.from_snapshot(original)
    candidate.wake()
    before = simulation_error(parent,simulator,evaluation_seeds,batch_size=batch_size)
    transcript, batches, unique = hashlib.sha256(), [], set()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    simulation_seconds, learning_seconds, seen = 0.0,0.0,0
    for epoch in range(epochs):
        iterator = iter(_batches(simulator,training_seeds,batch_size,epoch))
        while True:
            simulation_start = time.perf_counter()
            try: batch = next(iterator)
            except StopIteration: break
            events = [simulator.verify_transition(item['experience']) for item in batch]
            simulation_seconds += time.perf_counter()-simulation_start
            checkpoint_before = candidate.to_snapshot()['sha256']
            learning_start = time.perf_counter()
            for item,event in zip(batch,events):
                if event['experience_source'] != 'deterministic_simulation':
                    raise ValueError('simulation cannot masquerade as external observation')
                _reset(candidate)
                candidate.experience(Experience(event['observation'],target=event['target'],
                                                source=event['experience_source'],tag=item['experience']['sha256']))
                encoded = json.dumps(item['experience'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                transcript.update(encoded+b'\n')
                unique.add(snapshot_payload({'observation':event['observation'],'target':event['target']})['sha256'])
                seen += 1
            learning_seconds += time.perf_counter()-learning_start
            batches.append({'epoch':epoch,'transitions':len(batch),
                            'learner_checkpoint_before':checkpoint_before,
                            'learner_checkpoint_after':candidate.to_snapshot()['sha256']})
    elapsed, cpu = time.perf_counter()-wall_start, time.process_time()-cpu_start
    learned = candidate.to_snapshot()
    restored = AXMBrain.from_snapshot(learned)
    after = simulation_error(candidate,simulator,evaluation_seeds,batch_size=batch_size)
    retained = simulation_error(restored,simulator,evaluation_seeds,batch_size=batch_size)
    if parent.to_snapshot() != original: raise RuntimeError('simulation modified the parent')
    if seen != training_count: raise RuntimeError('simulation transition accounting mismatch')
    body = {'schema':'axm.micro-learning/v1','simulator_id':simulator.simulator_id,
            'simulator_version':simulator.version,'simulator_backend':simulator.backend,
            'rules':simulator.snapshot(simulator.reset(training_seeds[0]))['body']['rules'],
            'experience_source':'deterministic_simulation', 'training_seeds':list(training_seeds),
            'evaluation_seeds':list(evaluation_seeds),'epochs':epochs,'batch_size':batch_size,
            'parent':original,'candidate':learned,'experience_trace_sha256':transcript.hexdigest(),
            'batches':batches,'training_transitions':seen,
            'completed_training_episodes':len(training_seeds)*epochs,
            'unique_training_experiences':len(unique),'duplicate_experiences':seen-len(unique),
            'held_out_before':before,'held_out_after':after,'held_out_after_restore':retained,
            'parent_unchanged':True,'restore_exact':restored.to_snapshot()==learned,
            'timing':{'wall_seconds':elapsed,'cpu_seconds':cpu,'simulation_and_verification_seconds':simulation_seconds,
                      'learning_and_trace_seconds':learning_seconds,'transitions_per_second':seen/elapsed},
            'memory_measurement':None,'persistent_bytes_written_by_library':0,
            'adoption_state':'archive-only','automatic_adoption':False,
            'limits':['Numeric reference dynamics only; simulation is not external-world evidence.',
                      'Host-defined fixed action curriculum; learned simulation selection is not implemented.',
                      'Timings include receipt generation and integrity checks; no RAM/GPU measurement.']}
    return snapshot_payload(body)
