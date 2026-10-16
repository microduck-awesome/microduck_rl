# Roulade 原配方续训对照专职台账

当前：control64及原两条件末评全部完成，未合格，无active PID，GPU1释放；原PID160841/session31755已exit0。实时状态 `/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/roulade-control-pipeline-status.json`。根代理独占GPU0的teacher臂。

本次是根代理新明确授权的有限比较预算：两臂各64更新、N16384/T24，每臂25,165,824 controls，合计50,331,648；Roulade全部历史上限因此变为150,994,944。旧100,663,296步失败预算完整保留，不重置、不自动再延长。

共同父：`roulade-endpoint-allocation-001/checkpoint_000256.pt`，clock6144、LR0.0005852766346593505。Control环境/奖励/课程/物理/测量/actor/critic/PPO全部原样，来源为真实detached `roulade-control-worktree`；106个父数值源复制后只有trainer发生显式恢复/验证差分。训练参数、CAD和冻结v5不改。

父checkpoint只有一个活动CUDA随机流tensor，明确来源cuda:0；control将同一字节流映射cuda:1，CPU随机流保持原字节，其他卡流不访问/不改。实际actor/critic/normalizer/Adam、原LR与clock在恢复后逐位检查，再原生新episode reset；不把新环境重建说成完整物理轨迹回放。同卡旧路径与错误provenance由CPU契约检查，真实cuda1 draws再与独立同状态native generator逐位比较。

准入串联：CPU逐字段cfg/PPO比较→真实RNG映射oracle→成熟64×5含32真实timeout/终止状态/ONNX→actual16384×16。两级均从原256，验证权重不继承。源/138资产SHA、feedback/nonfinite、PPO KL/clip与8种容量日志护栏保留。通过后写qualification并等待根代理含双资格SHA的pair marker；marker到达后同进程正式256→320，再原nominal/measured5秒评测，相机200Hz/20ms曝光和健康分母分开报告。

GroundPick端点探查已结束未合格：正式75,497,472保持，剩余42未执行。此Roulade控制预算不混入GroundPick；不部署、无轮滑/Spin。

计划机器记录：`/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/roulade-control-plan-001.json`。生产补丁：`/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/roulade-control-resume.patch`。

## 准入通过，等待根启动标记

CPU逐字段env/agent完全相等，6项RNG契约与真实cuda1 normal/uniform/randperm共771值逐位PASS；原cuda0活动流和恢复cuda1流SHA完全一致。成熟64×5：32真实timeout、terminal/partialreset通过、ONNX最大5.96e-07，fault/nonfinite0，contact421/4096。

actual16384×16：6,291,456验证controls、24,576真实timeout，fault/nonfinite0、contact107960/1048576、max pooledKL.013815/clip.185558；剔除首3更新steady median107,699/aggregate107,389 controls/s，含本phase建环境/16更新/保存的wall=70.587s（共享import另计）。106数值SHA与138资产一致。资格`roulade-control-qualification-001.json` SHA6bc546cfa4fbf334b1bf50fcfca302b489850be14857dd9650cbf8d00adf8d02。正式仍0，worker只等待根带双资格SHA的marker。

根为减少空档已授权teacher先开64；这是同父/同seed/同64对照，启动时刻无需一致，不宣称逐轨迹配对。

## 正式已启动

根代理核对双方资格并写PASS_BOTH_ARMS双SHA marker，control同进程重新精确恢复原256，首两更新实际258/320、786,432新增controls、clock6192。继承actor/critic/N/Adam及LR均exact，非validation272权重。后续实际进度直接读`roulade-control-allocation-001/status.json`，两条件5s末评自动接续；任何guard停并保留。

## 控制臂关闭

正式新增25,165,824 controls /64更新，原256→320；原Roulade100,663,296历史保留。完整formal phase（共享imports/准入另计）242.747s、103671.12controls/s；steady aggregate108,235.44/median109,033.51。98,304真实timeout，maxKL0.02054/clip0.21152，feedback/nonfinite0，无8类容量或外置guard命中。

原nominal/measured各256world×5s：standing完整/body/head均0/128，四midroll body/head均0/32，健康相机分母0，actor/N未变。未因继续PPO64自然得到落稳/全滚；teacher对照由根代理合并分析。两条件与原256保存初态的逐位检查在`roulade-control-closeout-001.json`，只断言各字段实际结果，不外推跨GPU整条轨迹配对。

保留原256及新320，GPU1释放，不自动加训。最终`roulade-control-closeout-001.json`/`roulade-control-agent-current.json`。
