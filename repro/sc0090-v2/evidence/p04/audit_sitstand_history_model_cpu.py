from pathlib import Path
import mujoco,json,hashlib
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-history-base-worktree';xml=W/'src/mjlab_microduck/custom_v2/assets/robot.xml'
m=mujoco.MjSpec.from_file(str(xml)).compile();cpu=json.loads((R/'sitstand-history-base-cpu-001.json').read_text());rows=[]
for v in cpu['curriculum_mapping']:
 if v['term']!='head_pose_range':continue
 for name,(lo,hi)in zip(('neck_pitch','head_pitch','head_yaw','head_roll'),v['stage']['ranges']):
  j=m.joint(name);home=float(m.qpos0[j.qposadr[0]]);jlo,jhi=map(float,j.range);absolute=[home+lo,home+hi];assert home==0 and jlo<=absolute[0]<=absolute[1]<=jhi,(name,home,absolute,j.range)
  rows.append(dict(joint=name,native_step=v['native_step'],historical_step=v['historical_step'],HOME=home,command_delta=[lo,hi],absolute_target=absolute,model_limits=[jlo,jhi],margin=[absolute[0]-jlo,jhi-absolute[1]],within_model_range=True))
passive=[m.joint(i).name for i in range(m.njnt) if m.joint(i).name.startswith('passive_') and 'backlash' in m.joint(i).name];assert not passive
run=json.loads((R/'sitstand-history-base-smoke-001/run.json').read_text());init=run['environment']['scene']['entities']['robot']['init_state']['joint_pos'];assert init=={'.*':0.},init
p=run['environment']['rewards']['head_pose_tracking']['params'];assert p=={'command_name':'head_pose','std':.5},p
out=dict(status='PASS',source_xml=str(xml),source_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),head_ranges=rows,default_joint_pos=init,passive_backlash_joints=passive,active_head_reward_params=p,head_semantics='fine_std defaults None; output-side zero backlash mask; existing64/actual physical head Gaussian oracle validates actual path without an extraGPU run',scope='All configured future ranges checked, including stages beyond this320 budget')
(R/'sitstand-history-model-limits-cpu.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
