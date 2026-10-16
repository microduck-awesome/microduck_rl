from pathlib import Path
import ast,json,torch
p=Path('src/mjlab_microduck/custom_v2/velstand_ppo.py')
node=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='bc_minibatches')
scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),scope)
rows=[]
for n in (1,511,512,1023,1024,1535,1536,589821):
 permutation=torch.randperm(n);rng=torch.get_rng_state().clone()
 batches=scope['bc_minibatches'](permutation,requested=4,minimum=512)
 sizes=[len(x) for x in batches]
 assert len(batches)<=4 and max(sizes)-min(sizes)<=1
 assert min(sizes)>=min(n,512)
 assert torch.equal(torch.cat(batches),permutation)
 assert torch.equal(torch.cat(batches).sort().values,torch.arange(n))
 assert torch.equal(rng,torch.get_rng_state())
 rows.append(dict(samples=n,batch_sizes=sizes))
out=Path('../velstand-balanced-batch-cpu.json');assert not out.exists()
out.write_text(json.dumps(dict(status='PASS',checks=rows,scope='Every permutation element exactly once, balanced minimum-sized batches, at most4, RNG unchanged'),indent=2)+'\n')
print(rows)
