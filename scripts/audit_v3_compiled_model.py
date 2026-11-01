"""Compare V3 assembly MJCF with the actual compiled training entity on CPU."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'src/mjlab_microduck/custom_v2/assets/robot.xml'
    source=mujoco.MjModel.from_xml_path(str(path))
    cfg=load_env_cfg('Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline',play=True)
    cfg.scene.num_envs=1
    env=ManagerBasedRlEnv(cfg,device='cpu')
    mismatches=[]
    counts={}
    try:
        actual=env.sim.mj_model
        for field in ('timestep','integrator','iterations','ls_iterations','tolerance','cone','gravity'):
            a=np.asarray(getattr(source.opt,field));b=np.asarray(getattr(actual.opt,field))
            if not np.allclose(a,b,rtol=1e-7,atol=1e-10):
                mismatches.append({'kind':'option','field':field,'source':a.tolist(),'actual':b.tolist()})
        sections={
            'body':('body_parentid','body_mass','body_ipos','body_iquat','body_inertia','body_pos','body_quat'),
            'joint':('jnt_type','jnt_pos','jnt_axis','jnt_range','jnt_limited'),
            'geom':('geom_type','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity',
                    'geom_condim','geom_friction','geom_solref','geom_solimp','geom_priority'),
            'site':('site_pos','site_quat','site_size'),
        }
        sizes={'body':'nbody','joint':'njnt','geom':'ngeom','site':'nsite'}
        for kind,fields in sections.items():
            counts[kind]=0
            expected=set()
            for i in range(getattr(source,sizes[kind])):
                item=getattr(source,kind)(i)
                if not item.name:
                    mismatches.append({'kind':kind,'index':i,'field':'unnamed'})
                    continue
                if kind=='body' and item.name=='world':
                    continue
                name='robot/'+item.name
                expected.add(name)
                try:
                    idx=getattr(actual,kind)(name).id
                except KeyError:
                    mismatches.append({'kind':kind,'name':item.name,'field':'missing'})
                    continue
                counts[kind]+=1
                for field in fields:
                    a=np.asarray(getattr(source,field)[i]); b=np.asarray(getattr(actual,field)[idx])
                    if field=='body_parentid':
                        parent_src=source.body(int(a)).name
                        parent_dst=actual.body(int(b)).name.removeprefix('robot/')
                        if parent_src!=parent_dst:
                            mismatches.append({'kind':kind,'name':item.name,'field':field,
                                               'source':parent_src,'actual':parent_dst})
                    elif not np.allclose(a,b,rtol=1e-7,atol=1e-10):
                        mismatches.append({'kind':kind,'name':item.name,'field':field,
                                           'source':a.tolist(),'actual':b.tolist()})
                if kind=='geom' and source.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
                    src_name=source.mesh(int(source.geom_dataid[i])).name
                    dst_name=actual.mesh(int(actual.geom_dataid[idx])).name.removeprefix('robot/')
                    if src_name!=dst_name:
                        mismatches.append({'kind':kind,'name':item.name,'field':'mesh_binding',
                                           'source':src_name,'actual':dst_name})
            actual_names={getattr(actual,kind)(i).name for i in range(getattr(actual,sizes[kind]))
                          if getattr(actual,kind)(i).name.startswith('robot/')}
            extra=sorted(actual_names-expected)
            if extra:
                mismatches.append({'kind':kind,'field':'extra_robot_names','actual':extra})
        counts['mesh']=0
        for i in range(source.nmesh):
            name=source.mesh(i).name
            if not name:
                mismatches.append({'kind':'mesh','index':i,'field':'unnamed'})
                continue
            try:
                idx=actual.mesh('robot/'+name).id
            except KeyError:
                mismatches.append({'kind':'mesh','name':name,'field':'missing'})
                continue
            counts['mesh']+=1
            for field,offset,count in [('mesh_vert','mesh_vertadr','mesh_vertnum'),
                                       ('mesh_face','mesh_faceadr','mesh_facenum')]:
                start=int(getattr(source,offset)[i]);end=start+int(getattr(source,count)[i])
                other=int(getattr(actual,offset)[idx]);other_end=other+int(getattr(actual,count)[idx])
                a=getattr(source,field)[start:end];b=getattr(actual,field)[other:other_end]
                if a.shape!=b.shape or not np.allclose(a,b,rtol=1e-7,atol=1e-10):
                    mismatches.append({'kind':'mesh','name':name,'field':field,
                                       'source_shape':a.shape,'actual_shape':b.shape})
        report={'status':'PASS' if not mismatches else 'MISMATCH',
                'protocol':'CPU compile of registered paired-error standup task; compare named robot entity to raw V3 assembly model before per-environment randomization',
                'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'entity_counts':counts,
                'source_total_mass_kg':float(source.body_mass.sum()),
                'compiled_robot_total_mass_kg':float(sum(actual.body('robot/'+source.body(i).name).mass[0]
                                                          for i in range(1,source.nbody))),
                'physics_timestep_s':float(actual.opt.timestep),'control_period_s':env.step_dt,
                'mismatches':mismatches,
                'limitation':'Named body/joint/geom/site fields and mesh bindings, vertices and faces only; no claim about real robot, randomization or complete dynamic parity.'}
        output=root/'docs/v3-compiled-model-audit.json'
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        if mismatches:raise SystemExit(1)
    finally:env.close()


if __name__=='__main__':main()
