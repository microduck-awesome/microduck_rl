"""Materialize each historical source version; only replace the collision XML/add its meshes."""
from pathlib import Path
import json,hashlib,tarfile,os,shutil,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT.parent
DEST=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762/sources')
COLLISION=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261023/rise3762-collision/source/src/mjlab_microduck/custom_v2/assets')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(n):return(n.tag,sorted(n.attrib.items()),(n.text or '').strip(),tuple(canonical(c)for c in n))
inv=json.loads((ROOT/'history-inventory.json').read_text());snap=json.loads((PACKAGE/'snapshots/rise_baseline_paired3762.json').read_text())
xmlname='src/mjlab_microduck/custom_v2/assets/robot.xml';h=snap['files'][xmlname]['sha256'];original=ET.fromstring((PACKAGE/'objects'/h[:2]/h).read_bytes());corrected=ET.parse(COLLISION/'robot.xml').getroot();added=[]
for parent in corrected.iter():
 for n in list(parent):
  if n.get('name','').startswith('sim2real_shell_'):added.append(n);parent.remove(n)
assert canonical(original)==canonical(corrected)
assert sum(x.tag=='mesh'for x in added)==sum(x.tag=='geom'for x in added)==32
assert original.find('compiler').get('inertiafromgeom')=='false'
DEST.mkdir(exist_ok=False);cache=DEST/'.static-cache';cache.mkdir();static={}
for name,entry in snap['files'].items():
 p=Path(name)
 if name.startswith('src/') and p.suffix not in ('.py','.cpp') or name in ('AGENTS.md','LICENSE','LICENSE-MODELS.md','pyproject.toml','uv.lock') or name.startswith('licenses/'):
  h=entry['sha256'];src=PACKAGE/'objects'/h[:2]/h;assert sha(src)==h
  target=cache/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);target.chmod(0o755 if entry['executable'] else 0o644);static[name]=h
rows=[]
for phase in inv['phases']:
 dest=DEST/phase['name'];dest.mkdir()
 for name in static:
  p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);os.link(cache/name,p)
 with tarfile.open(ROOT/'evidence'/phase['name']/'source.tar.gz')as tar:
  for m in tar.getmembers():
   assert m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts
   p=dest/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read());p.chmod(m.mode&0o777);os.utime(p,(m.mtime,m.mtime))
 target=dest/xmlname;temp=target.with_suffix('.replacement');shutil.copyfile(COLLISION/'robot.xml',temp);temp.replace(target)
 additions={}
 for src in sorted((COLLISION/'assets').glob('sim2real_shell_*.stl')):
  p=dest/'src/mjlab_microduck/custom_v2/assets/assets'/src.name;shutil.copy2(src,p);additions[str(p.relative_to(dest))]=sha(p)
 assert len(additions)==32
 numeric=sorted([*dest.glob('src/**/*.py'),*dest.glob('src/**/*.cpp'),dest/'scripts/train_v2_standup.py',dest/'scripts/v2_python.sh'])
 actual={str(p.relative_to(dest)):sha(p)for p in numeric};assert actual==phase['source_files'],phase['name']
 for name,h in static.items():assert sha(dest/name)==(sha(COLLISION/'robot.xml')if name==xmlname else h),(phase['name'],name)
 rows.append(dict(phase=phase['name'],path=str(dest),numerical_source_exact=True,numerical_files=len(actual),static_files=len(static),only_modified_static_file=xmlname,original_xml_sha256=phase['historical_preflight']['robot_xml_sha256'],corrected_xml_sha256=sha(dest/xmlname),added_meshes=additions))
(DEST/'.gitignore').write_text('.static-cache/\n__pycache__/\n*.pyc\nMUJOCO_LOG.TXT\n')
(ROOT/'materialized-sources.json').write_text(json.dumps(dict(status='PASS',collision_original_elements_exact=True,static_file_hashes=static,phases=rows),indent=2)+'\n')
print(json.dumps(dict(status='PASS',phases=len(rows),static_files=len(static),only_difference='robot.xml + 32 new shell meshes')))
