# GroundPick 健康低位曝光与完整历史课程

当前：固定fresh512和原12s双条件四节点末评已全部完成，正式301,989,888新增、Ground全历史377,487,360；预算关闭，无自动续训。最终原门槛曲线及头稳/接触分母见本文末及`groundpick-health-history-final-review.json`。

批准固定一条从零谱系：N24576/T24/E5/M24×512，新增301,989,888 controls、61,440次原生Adam；既有75,497,472保留，全Ground历史上限377,487,360。不继承旧128actor/N/Adam或验证权重，不中途换配方。每64保留检查点，512后统一128/256/384/512两条件原12s完整周期评测，无自动第二段。

课程step÷6，参考N4096/T24/M4单minibatch24,576保持；旧max48,000映射8,000，终nativeclock12,288。原strict>且20s=1,000control自然reset，所以实际升级需8000后首reset（如全20s整周期为9000）；逐节点记录真实manager参数与reward权重，不以clock算式冒称生效。4s物理相位/.2s低位/20s episode及载荷时间不缩。

仅两项必要语义变化：

- 25% reset用008的**desired physical q**及原root+2mm/±3mrad扰动，与同mask φ=.375绑定；75%保留原HOME和随机相位。新reset-only command子类继承原compute/编码/周期，只在原native reset抽完相位后覆盖选中低位。原观测仍真实S0/O0，actor不增mask/真值，不提供补偿command/关节模板奖励/teacher。旧类/task不改。
- proximity本身非负，head/self found即正奖0；orientation可负，采用`min(raw,0)+clean*max(raw,0)`保留负部。clean只用同reward/reader边界的已有head/self found，不加速度门槛或额外税。原1N头冲击成本和其余原奖励/物理全保留。新task独立oracle显式采用同符号语义，旧task原oracle不变。

新terminal类白名单条目并非已准入。CPU先验证phase/mask部分reset、compute继承和shadow独立；GPU须实际低位reset的q/phase/rawactor/payload时序、真实partialreset、phase .375/.420/.425/.470/.995等fast/full successor对照及真正timeout。补200Hz接触只读记录，旧008报告仅50Hz接触边界，不改写成全200Hz证明。

008 fixed-command端点是物理证据，非策略：nominal0/30g各32/32，measured空29/32+30g32/32；3条只mouth>25mm，原gate保留。0/30g证据不等同原训练10–40g全部通过。低位φ=.375原payload力0，.425后原ramp加载，初始明确初始化payload，不能在reset额外造载荷/清反馈。

保存数据即时分项审计：002头撑φ=.375平均力.729N，嘴奖2.9569+朝向1.9961，新门控均0；008实际φ=.6两项约1.5218+1.06375，clean全1且原样保留。仅两分项反事实，不是完整return或学习因果证明。证据`groundpick-health-history-saved-reward-audit.json`。

CPU001外置diff路径整数键串接失败、002原生terrain closure对象身份不同，均0GPU/0PPO，原件保留；003外置核验闭包源码/值等价，未为了报告修改数值源。具体cfg/源码SHA、patch、预算见`groundpick-health-history-plan-001.json`。吞吐如实报告，100k不是唯一正确性阻挡，feedback/非有限/容量/PPO护栏保留。

CPU资格实测：4096首次+241局部reset，HOME3268/LOW1069；部分reset其它q/root/phase/mask逐位不变，lowmask与φ=.375绑定，compute原样继承，CPUshadow相位无live回写。283资产SHAexact、terrain工厂closure源码/闭包值exact。CPU外置闭包指纹缺空格另留003失败与修正前脚本，成功日志004；数值源未为报告问题修改。

实际GPU资格：32world中7low，真实96诊断controls检查native phase/rawactor/payload与200Hz接触记录，6phase×fast/full共12 successor cases和partial reset通过；不把接触采样本身宣称无接触。64×5为32真实timeout/ONNX2.38e−7，actual24576×16为9,437,184验证controls、12,288timeout，五epochs×24实际1,920Adam。稳态median86,524.24/aggregate86,236.77 controls/s，未达10万，不追加性能修改。完整细值与fresh权重/N/Adam/clock0证据见 `groundpick-health-history-admission-receipt.json`。

## 固定512与原协议四节点最终曲线

| 节点 | 条件 | 3周期低位/各256 | 3周期回站/各256 | 完整/各256 | fault/各256 |
|---:|---|---|---|---|---|
|128|nominal|[0, 0, 0]|[234, 248, 246]|[0, 0, 0]|[0, 0, 0]|
|128|measured|[0, 0, 0]|[229, 243, 243]|[0, 0, 0]|[0, 0, 0]|
|256|nominal|[0, 0, 0]|[234, 248, 249]|[0, 0, 0]|[0, 0, 0]|
|256|measured|[0, 0, 0]|[232, 248, 249]|[0, 0, 0]|[0, 0, 0]|
|384|nominal|[0, 0, 0]|[233, 244, 244]|[0, 0, 0]|[0, 0, 0]|
|384|measured|[0, 0, 0]|[222, 236, 233]|[0, 0, 0]|[0, 0, 0]|
|512|nominal|[0, 0, 0]|[227, 244, 245]|[0, 0, 0]|[0, 0, 0]|
|512|measured|[0, 0, 0]|[199, 214, 208]|[0, 0, 0]|[0, 0, 0]|

正式新增301,989,888/实际61,440 Adam，Ground历史377,487,360，预算关闭不自动续训。完整formal阶段3456.926s/87357.94controls/s；稳态median89602.28/aggregate87910.59。worker含资格/等待/8末评另列4193.851s。

各组四载荷/推扰、原gate parity、初态配置逐位核验、200Hz额外head/self筛查及相机健康分母见`groundpick-health-history-final-review.json`。不以静态008或训练reset命中替代完整周期能力，所有候选保留、实机未验证。
