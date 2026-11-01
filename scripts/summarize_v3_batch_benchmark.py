"""Summarize completed GPU 0 batch preflights without claiming convergence."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/l40-vm01/data2/experiments/v3-rise6804-20261101')
OUT=Path(__file__).resolve().parents[1]/'docs/v3-batch-preflight.json'


def main():
    rows=[]
    for n in (8192,16384,24576):
        path=ROOT/f'batch{n}-report.json'
        report=json.loads(path.read_text())
        assert report['status']=='PASS' and report['envs']==n and report['iterations']==13
        assert report['throughput']['measured_updates']==10 and report['physics_cuda_graph']
        assert report['steps_per_env']==24 and report['learning_epochs']==5 and report['minibatches']==4
        csv_path=ROOT/f'batch{n}-gpu-samples.csv'
        gpu_peak=None
        if csv_path.exists():
            samples=[r for r in csv.reader(csv_path.open()) if len(r)==4 and r[1].strip()=='0']
            assert samples
            gpu_peak=max(int(r[2].strip()) for r in samples)
        rows.append({'envs':n,'steps_per_update':n*24,'minibatch_samples':n*24//4,
                     'median_update_s':report['throughput']['median_wall_s'],
                     'median_control_steps_per_s':report['throughput']['median_end_to_end_steps_per_s'],
                     'sampled_gpu0_peak_mib':gpu_peak,
                     'max_contacts':report['peak_contacts_across_worlds'],
                     'contact_capacity':report['contact_capacity_across_worlds'],
                     'onnx_native_max_error':report['onnx_cpu_inference']['normalizer_parity_max_error'],
                     'report':str(path),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'gpu_samples':str(csv_path) if csv_path.exists() else None,
                     'gpu_samples_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest() if csv_path.exists() else None})
    for row in rows:
        row['relative_control_throughput']=row['median_control_steps_per_s']/rows[0]['median_control_steps_per_s']
        row['relative_update_time']=row['median_update_s']/rows[0]['median_update_s']
    result={'status':'PASS','device':'physical GPU 0','warmup_updates':3,'measured_updates':10,
            'rows':rows,'interpretation':'Short fresh-policy throughput only; environment count changes PPO minibatch and samples per update. Speed is not convergence or trained-policy equivalence. GPU memory is sampled, not an allocator peak; 8192 was not sampled.'}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
