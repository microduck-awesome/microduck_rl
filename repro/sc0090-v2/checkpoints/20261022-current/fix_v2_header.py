"""Repackage the two immutable snapshots with the pinned firmware V2 converter."""
from pathlib import Path
import hashlib,json,struct,sys,zlib
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'tools'))
import convert_policy,v2_contract

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 report=[]
 for skill,n,runtime in [('rise',4000,v2_contract.RUNTIME_ID),('walk',4600,v2_contract.WALKING_RUNTIME_ID)]:
  d=root/f'{skill}-{n}';target=d/f'{skill}-{n}.mdp';old=target.read_bytes();onnx_before=sha((d/'policy.onnx').read_bytes());checkpoint_before=sha((d/'checkpoint.pt').read_bytes())
  old_version=struct.unpack_from('<I',old,8)[0]
  if old_version==1:
   try:v2_contract.info(old)
   except ValueError:pass
   else:raise AssertionError('Old version1 must not pass the V2 decoder')
   (d/'legacy-version1-converter.json').write_bytes((d/'converter-original.json').read_bytes())
  result=convert_policy.export(d/'policy.onnx',target);new=target.read_bytes();info=v2_contract.info(new)
  assert struct.unpack_from('<I',new,8)[0]==2
  assert struct.unpack_from('<I',new,84)[0]==runtime
  assert old[:8]==new[:8]and old[12:84]==new[12:84]and old[128:]==new[128:]
  assert struct.unpack_from('<I',new,28)[0]==zlib.crc32(new[128:])
  assert new[52:84]==bytes.fromhex(onnx_before)
  assert sha((d/'policy.onnx').read_bytes())==onnx_before and sha((d/'checkpoint.pt').read_bytes())==checkpoint_before
  assert new[84:128]==v2_contract.extension(result['source_metadata'],onnx_before)
  changed_metadata={**result['source_metadata'],'action_scale':'0.5'}
  try:v2_contract.extension(changed_metadata,onnx_before)
  except ValueError:pass
  else:raise AssertionError('Candidate metadata guard must remain strict')
  (d/'converter-original.json').write_text(json.dumps(result,indent=2)+'\n')
  result.update(model_id=f'v2_sim2real_{skill}_{n}_20261022',runtime_contract='runtime-contract.json',checkpoint_sha256=checkpoint_before,hardware_tested=False)
  (d/f'{skill}-{n}.json').write_text(json.dumps(result,indent=2)+'\n')
  p=d/'runtime-contract.json';contract=json.loads(p.read_text());contract['training_runtime_family']=contract.pop('runtime_family',contract.get('training_runtime_family'))
  contract.update(deployment_runtime_family=info['runtime'],policy_runtime_id_hex=f'0x{runtime:08X}',mdp_header_version=2,deployment_contract=result['deployment_contract'],deployment_contract_sha256=info['deployment_contract_sha256']);p.write_text(json.dumps(contract,indent=2)+'\n')
  validation=dict(model=d.name,previous_header_version=old_version,header_version=2,policy_runtime_id_hex=f'0x{runtime:08X}',old_mdp_sha256=sha(old),new_mdp_sha256=sha(new),coefficient_payload_sha256=sha(new[128:]),coefficient_payload_unchanged=True,onnx_sha256=onnx_before,checkpoint_sha256=checkpoint_before,weights_normalizer_and_onnx_unchanged=True,crc_verified=True,source_sha_verified=True,pinned_v2_decoder_passed=True,exact_candidate_guard_preserved=True,header_extension=info)
  p=d/'validation.json';v=json.loads(p.read_text());v['v2_header']=validation;v['converter_source_sha256']=sha((root/'tools/convert_policy.py').read_bytes());p.write_text(json.dumps(v,indent=2)+'\n');report.append(validation)
 (root/'v2-header-fix.json').write_text(json.dumps(dict(firmware_source_revision='13eea5de53224b5cb6bd1bf0de91ff4f1f5692de',scope='Only version field and44-byte contract extension changed in MDP binaries; no network or firmware change',models=report),indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
