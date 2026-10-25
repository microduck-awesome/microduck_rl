"""Compare effective collision settings after the official Entity builders."""
import json
from pathlib import Path
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg

def describe(task):
    cfg=load_env_cfg(task)
    model=cfg.scene.entities['robot'].build().spec.compile()
    rows=[]
    for i in range(model.ngeom):
        if model.geom_contype[i] or model.geom_conaffinity[i]:
            rows.append(dict(name=model.geom(i).name,body=model.body(int(model.geom_bodyid[i])).name,
                type=int(model.geom_type[i]),contype=int(model.geom_contype[i]),conaffinity=int(model.geom_conaffinity[i]),
                condim=int(model.geom_condim[i]),priority=int(model.geom_priority[i]),friction=model.geom_friction[i].tolist(),
                solref=model.geom_solref[i].tolist(),solimp=model.geom_solimp[i].tolist()))
    return dict(active_geoms=rows,explicit_pairs=model.npair,exclude_signatures=model.exclude_signature.tolist())

out={name:describe(task) for name,task in [('official','Mjlab-Velocity-Flat-MicroDuck'),('v2','Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline')]}
p=Path('docs/evidence/official-walk/collision-comparison.json');p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
