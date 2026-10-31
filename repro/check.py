"""Read-only source/runtime audit. Does not create an environment or train."""
import argparse, hashlib, importlib.metadata as metadata, json, platform, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment', action='store_true', help='Also verify installed package versions and BAM commit')
    p.add_argument('--gpu', action='store_true', help='Also import torch and report CUDA availability')
    p.add_argument('--output', type=Path)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'repro/environment.json').read_text())
    errors=[]
    for name,digest in expected['source_sha256'].items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest: errors.append('source: '+name)
    report={'profile':expected['profile'],'source_files_checked':len(expected['source_sha256']),'python':platform.python_version(),'errors':errors}
    if a.environment:
        if not platform.python_version().startswith(expected['python']+'.'): errors.append('Python must be '+expected['python'])
        installed={}
        try:
            project=metadata.distribution('mjlab-microduck')
            if not any(e.group=='mjlab.tasks' and e.name=='mjlab_microduck' for e in project.entry_points): errors.append('Missing mjlab task entry point')
        except metadata.PackageNotFoundError: errors.append('Install this project with --no-deps --editable . to register its tasks')
        for name,version in expected['dependencies'].items():
            try: installed[name]=metadata.version(name)
            except metadata.PackageNotFoundError: installed[name]=None
            if installed[name]!=version: errors.append(f'{name}: expected {version}, got {installed[name]}')
        try: direct=metadata.distribution('better-actuator-models').read_text('direct_url.json') or ''
        except metadata.PackageNotFoundError: direct=''
        if expected['bam_commit'] not in direct: errors.append('BAM VCS commit does not match')
        report['dependencies']=installed
        if (root/'repro/requirements-runtime.txt').exists():
            from packaging.requirements import Requirement
            pinned=0
            for line in (root/'repro/requirements-runtime.txt').read_text().splitlines():
                line=line.strip()
                if not line or line.startswith('#'): continue
                req=Requirement(line)
                if req.marker and not req.marker.evaluate(): continue
                try: version=metadata.version(req.name)
                except metadata.PackageNotFoundError:
                    errors.append('Missing runtime requirement: '+req.name);continue
                if req.specifier and version not in req.specifier: errors.append('Runtime pin mismatch: '+str(req))
                pinned+=1
            report['runtime_requirements_checked']=pinned
        else:
            import tomllib
            lock=tomllib.loads((root/'uv.lock').read_text());locked={}
            for package in lock['package']: locked.setdefault(package['name'],set()).add(package['version'])
            checked=0
            for name,versions in locked.items():
                if name=='mjlab-microduck': continue
                try: version=metadata.version(name)
                except metadata.PackageNotFoundError: continue
                if version not in versions: errors.append('Installed version differs from uv.lock: '+name)
                checked+=1
            report['installed_lock_packages_checked']=checked
    if a.gpu:
        import torch
        report['cuda']={'torch_cuda':torch.version.cuda,'available':torch.cuda.is_available(),'devices':torch.cuda.device_count()}
        report['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)
        minimum=2 if expected['profile']=='rise6804' else 1
        if torch.cuda.device_count()<minimum: errors.append(f'{minimum} CUDA GPUs required')
    report['status']='FAIL' if errors else 'PASS'
    result=json.dumps(report,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(result)
    print(result,end='')
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
