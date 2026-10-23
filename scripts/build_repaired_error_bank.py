"""Build nonoverlapping paired q/v error blocks from the repaired model audit."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/repaired-measured-onpolicy-errors-1024')
OUT=Path('src/mjlab_microduck/custom_v2/training-assets/repaired-paired-errors-v2')
OUT.mkdir(parents=True,exist_ok=False)
blocks=[];sources={}
for seed in (2026101501,2026101502):
    path=ROOT/f'paired-errors-{seed}.npz'
    with np.load(path) as data:
        values=data['samples'];assert values.shape==(400,224,14,6)
        assert list(data['channels'][:2])==['q_error_at_query','v_error_at_query']
        for start in range(16,385,16):
            block=values[start:start+16,:,:,:2].transpose(1,2,0,3).reshape(-1,16,2)
            assert block.shape==(224*14,16,2);blocks.append(block)
    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
bank=np.concatenate(blocks).astype(np.float32);assert np.isfinite(bank).all()
np.savez_compressed(OUT/'error-blocks.npz',blocks=bank)
report=dict(schema=1,id='repaired-S0-O0-onpolicy-paired-errors-v2',bank_sha256=hashlib.sha256((OUT/'error-blocks.npz').read_bytes()).hexdigest(),
 shape=list(bank.shape),sample_dt_s=.02,query_delay_s=.02,voltage_v=[10.2,11.1,12.4],channels=['physical_position_error_rad','physical_velocity_error_rad_s'],
 source_kind='Frozen P04-1024 successful recovery in repaired S0/O0 measured model; synthetic, no new hardware data',
 source_report_sha256=hashlib.sha256((ROOT/'report.json').read_bytes()).hexdigest(),sources=sources,
 limitations=['Development seeds used for model-error design; not held-out validation',
 'Seven zero-command recovery pose families, no dedicated commanded-head traces',
 'Pool joints and voltages; no online q/v/load conditional selection',
 'Within-block temporal and paired q/v correlation retained; cross-block continuity and faults not reproduced',
 'Approximate baseline randomization, never equivalent to full S0/O0/R1'],
 measurement_role='Baseline empirical error randomization; all target qualification still uses actual repaired S0/O0',
 builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'manifest.json').write_text(json.dumps(report,indent=2));print(report)
