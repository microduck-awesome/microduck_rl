"""Render saved physics positions; no new control or simulation integration."""
from pathlib import Path
import argparse,math
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1];spec=mujoco.MjSpec.from_file(str(root/'src/mjlab_microduck/custom_v2/assets/robot.xml'))
spec.worldbody.add_geom(name='replay_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2,2,.1],rgba=[.8,.82,.85,1])
spec.worldbody.add_light(pos=[0,-1,2],dir=[0,0,-1])
m=spec.compile();d=mujoco.MjData(m);data=np.load(a.trace);qpos=data['qpos'];origins=data['env_origins'];assert qpos.shape[1]==7 and qpos.shape[2]==m.nq
names=['standing','sitting','prone','frontier','left_side','right_side','supine']
w,h=320,240;renderer=mujoco.Renderer(m,height=h,width=w);cams=[]
for j in range(7):
 q=qpos[0,j,3:7];yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))
 cam=mujoco.MjvCamera();cam.distance=.48;cam.azimuth=math.degrees(yaw)+130;cam.elevation=-18;cam.lookat[:]=[0,0,.10];cams.append(cam)
def tile(i,j):
 d.qpos[:]=qpos[i,j];d.qpos[:3]-=origins[j];mujoco.mj_forward(m,d);cams[j].lookat[:2]=d.qpos[:2];renderer.update_scene(d,camera=cams[j]);im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im);draw.rectangle((0,0,w,18),fill='white');draw.text((5,3),f'{names[j]}   {i*.02:.2f}s',fill='black');return im
selected=[0,50,100,200,400];strip=Image.new('RGB',(7*w,len(selected)*h),'white')
for row,i in enumerate(selected):
 for j in range(7):strip.paste(tile(min(i,len(qpos)-1),j),(j*w,row*h))
strip.save(a.output/'recovery-filmstrip.png')
with imageio.get_writer(a.output/'recovery-seven-poses.mp4',fps=25,codec='libx264',quality=8)as writer:
 for i in range(0,len(qpos),2):
  canvas=Image.new('RGB',(3*w,3*h),'white')
  for j in range(7):canvas.paste(tile(i,j),((j%3)*w,(j//3)*h))
  draw=ImageDraw.Draw(canvas);draw.text((w+15,2*h+35),'New collision model\nSaved training-physics replay\nIllustration: one trial per pose\nSee full 1792-trial report separately',fill='black')
  writer.append_data(np.asarray(canvas))
renderer.close();print(a.output)
