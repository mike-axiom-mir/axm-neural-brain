"""Host CLI for direct simulator-to-brain evidence, with retained checkpoints."""
import argparse
import json
from pathlib import Path
import platform

from neural.axm_brain import AXMBrain,BrainConfig
from neural.axm_brain.simulation import learn_in_simulation
from axm_neural_network.microsim import MicroDynamics
from neural.axm_brain.state import snapshot_payload


def run():
    runs=[]
    for seed in (17,41,73):
        brain=AXMBrain(BrainConfig(3,8,2,seed=seed,learning_rate=.04,replay_capacity=32))
        receipt=learn_in_simulation(brain,MicroDynamics(),range(12),range(100,112),epochs=6)
        runs.append(receipt)
    return {'schema':'axm.direct-simulation-evidence/v1','python':platform.python_version(),
            'platform':platform.system(),'runs':runs,
            'limits':['Numeric micro-dynamics only, fixed host curriculum, no external-world or UC claim.',
                      'All three brain seeds use the same declared training/evaluation simulator distributions.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full',action='store_true',help='Include complete checkpoints and every batch record')
    args=parser.parse_args()
    data=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.full:
        exported=data
    else:
        exported={k:v for k,v in data.items() if k!='runs'}
        exported['format']='compact evidence; use --full to retain complete checkpoints and batch records'
        exported['runs']=[]
        for receipt in data['runs']:
            b=dict(receipt['body'])
            for key in ('parent','candidate'):
                state=b.pop(key)
                b[key+'_sha256']=state['sha256']
                b[key+'_config']=state['body']['config']
            b['batch_trace_sha256']=snapshot_payload(b.pop('batches'))['sha256']
            exported['runs'].append({'body':b,'full_receipt_sha256':receipt['sha256']})
    args.output.write_text(json.dumps(exported,indent=2)+'\n')
    for r in data['runs']:
        b=r['body']
        print(json.dumps({'seed':b['parent']['body']['config']['seed'],'before':b['held_out_before']['mse'],
                          'after':b['held_out_after']['mse'],'retained':b['restore_exact'],
                          'training_transitions':b['training_transitions'],'timing':b['timing']}))
