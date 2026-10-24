"""Extract only ancestor-run launch commands from the original session record."""
from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
SESSION=Path('/home/l40-vm01/.codex/sessions/2026/10/15/rollout-2026-10-15T09-50-36-01a13d41-0df5-7443-a8a8-155b6a46523c.jsonl')
rows=json.loads((ROOT/'history-inventory.json').read_text())['phases'];names=[x['name']for x in rows];found={}
for line in SESSION.open():
 if not any(n in line for n in names):continue
 obj=json.loads(line);v=obj.get('payload',{});text=v.get('input','')
 if v.get('type')!='custom_tool_call':continue
 for match in re.finditer(r'(?:"cmd"|cmd)\s*:\s*("(?:[^"\\]|\\.)*")',text):
  cmd=json.loads(match.group(1))
  if 'scripts/train_v2_standup.py'not in cmd:continue
  for name in names:
   if name in found:continue
   if not re.search(r'--output\s+\S*'+re.escape(name)+r'(?=\s|$)',cmd):continue
   lines=[s for s in cmd.splitlines()if 'scripts/train_v2_standup.py' in s and '--output' in s and name in s]
   if not lines:continue
   found[name]=dict(timestamp=obj.get('timestamp'),session_id='01a13d41-0df5-7443-a8a8-155b6a46523c',tool_call_id=v.get('call_id'),launch_command=lines[-1],enclosing_tool_input_sha256=hashlib.sha256(text.encode()).hexdigest())
 if len(found)==len(names):break
out=dict(status='PASS' if len(found)==len(names)else'INCOMPLETE',commands=found,missing=sorted(set(names)-set(found)))
(ROOT/'historical-launch-evidence.json').write_text(json.dumps(out,indent=2)+'\n')
for name in names:print(name,found.get(name,{}).get('launch_command','MISSING'))
