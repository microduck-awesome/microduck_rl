> **2026-10-24 步行实机对比**：新增[三版新碰撞步行候选A3500/B3000/C3750](checkpoints/20261024-walk-three-model-hardware-test/README.md)，均为600.94g配方，含可独立下载的ONNX和Web上传MDP、checkpoint及差异说明。优先比较A；仍有低速欠响应、自碰/停止/模型限位问题，未作实机合格声明。

> **2026-10-23 正确映射重训候选**：新增[起身1400实机验证包](checkpoints/20261023-rise1400/README.md)，含Web上传MDP、ONNX、checkpoint、精确源码及评测。全量随机化起身达标；抗扰未毕业、头部明显偏姿，供用户授权实测。旧错误映射候选仍保持撤回。

> 本次交付补充：25个checkpoint全部在独立Release模型包，3个起身官方ONNX一并提供。以下保留原清单证据；原“未复制/未上传”说明仅描述清单生成时状态。

# 训练模型与评测清单

> **2026-10-22补充清单**：[12套历史起身V2实测候选](checkpoints/20261022-rise-comparison/README.md)及[机器可读模型表](checkpoints/20261022-rise-comparison/models.json)。新增3506/4018基础、3756/4012位置DR及头控分支，并提供三套既有参照的V2 MDP；下文25模型表保留历史身份。新版V2二进制不代表已完成新的实机验收。

此清单列出25个保留检查点。所有权重SHA均直接重算，所有源工作树均与各自训练记录逐文件一致。只记录路径、哈希和证据，不上传权重。全部模型尚无实机验收，未声明部署合格。

## 候选、失败与保留理由

| 模型 | 已记录谱系交互 | PPO Adam/参数 | 主要结论 |
|---|---:|---:|---|
| rise_p04_micro1856 | 1459617792 | 83200 | 实际P04恢复1792/1792；主动头精度仍不完整；负pitch全1024无先倒 |
| rise_baseline_paired3762 | 739639296 | 75240 | 实际P04恢复1792/1792；主动头精度仍不完整 |
| rise_baseline_to_p04_4466 | 1293287424 | 89320 | 实际P04恢复1792/1792；主动头精度仍不完整 |
| walking449 | 353107968 | 8980 | 动作/教师候选；零速跺脚、头稳与跟踪未合格 |
| walking512 | 402653184 | 10240 | 动作/教师候选；零速跺脚、头稳与跟踪未合格 |
| velstand256 | 553648128 | 15360 | nominal 336/336，measured 336/336存活；相机/头偏差/直行仍有取舍 |
| velstand_control320 | 591396864 | 16640 | nominal 336/336，measured 336/336存活；相机/头偏差/直行仍有取舍 |
| velstand_camera_mild320 | 591396864 | 16640 | nominal 335/336，measured 335/336存活；相机/头偏差/直行仍有取舍 |
| velstand_head_free320 | 591396864 | 16640 | nominal 335/336，measured 336/336存活；相机/头偏差/直行仍有取舍 |
| velstand_head_bias384 | 629145600 | 17920 | nominal 336/336，measured 335/336存活；相机/头偏差/直行仍有取舍 |
| velstand_head_bias448 | 666894336 | 19200 | nominal 336/336，measured 336/336存活；相机/头偏差/直行仍有取舍 |
| velstand_head_bias512 | 704643072 | 20480 | nominal 336/336，measured 336/336存活；相机/头偏差/直行仍有取舍 |
| velstand_head_bias576 | 742391808 | 21760 | nominal 335/336，measured 336/336存活；相机/头偏差/直行仍有取舍 |
| sitstand640 | 251658240 | 51200 | SIT尾保持约全过，STAND0/384；主动头0/672，两条件未合格 |
| sitstand_near_teacher768 | 301989888 | 61440 | SIT尾保持约全过，STAND0/384；主动头0/672，两条件未合格 |
| sitstand_near_teacher896 | 352321536 | 71680 | SIT尾保持约全过，STAND0/384；主动头0/672，两条件未合格 |
| sitstand_near_teacher960 | 377487360 | 76800 | SIT尾保持约全过，STAND0/384；主动头0/672，两条件未合格 |
| groundpick_history512 | 301989888 | 61440 | 原3周期完整数 nominal:0,0,0 / measured:0,0,0（每周期256）；未合格 |
| groundpick_precision704 | 415236096 | 84480 | 原3周期完整数 nominal:0,2,3 / measured:0,2,1（每周期256）；未合格 |
| groundpick_precision768 | 452984832 | 92160 | 原3周期完整数 nominal:0,5,0 / measured:0,2,0（每周期256）；未合格 |
| ballkick_health256 | 150994944 | 30720 | 原完整/加中性头 nominal:51/48 of256 / measured:44/41 of256；只右足、未全过 |
| ballkick_health384 | 226492416 | 46080 | 原完整/加中性头 nominal:44/20 of256 / measured:33/18 of256；只右足、未全过 |
| ballkick_precision384 | 226492416 | 46080 | 原完整/加中性头 nominal:67/39 of256 / measured:51/27 of256；只右足、未全过 |
| ballkick_precision512 | 301989888 | 61440 | 原完整/加中性头 nominal:28/8 of256 / measured:19/5 of256；只右足、未全过 |
| roulade1536 | 603979776 | 122880 | 两条件standing完整0/128，各midroll落稳0；完整历史剂量仍未合格 |

## 成本口径

复跑评测使用保留权重，新增PPO为0。重新训练必须包含所选谱系及独立教师前置训练，不能只报最后一段；JSON记录实际N/T/E/M、原生clock、PPO及BC Adam、父权重和teacher依赖。检查点controls有些已经继承warm-start控制量，不能再重复相加。完整谱系wall不齐全，明确为null；仅保留有范围定义的阶段计时，不用稳态吞吐估算总时间。

研究累计包括失败分支，不能当作重放唯一成功谱系所需成本：

| 技能 | 累计正式控制 |
|---|---:|
| rise_stand | 5,619,253,248 |
| walking | 402,653,184 |
| velstand | 415,236,096 |
| sitstand | 603,979,776 |
| groundpick | 528,482,304 |
| ballkick | 477,757,440 |
| roulade | 754,974,720 |

预检、物理诊断、独立评测和CPU准备不在上述正式控制总数内。研究累计wall暂无统一完整记录。旧schema13快照的Sit/Ground在生成本清单前已完成，采用相应最终closeout更新。

## 精确权重与源码

下列路径相对于 artifact_root：`/home/l40-vm01/data2/experiments/microduck-v2-20261015`。每项JSON同时给出绝对路径、source worktree、source.tar.gz SHA、run.json SHA、源清单SHA及评测报告SHA。

- **rise_p04_micro1856**：`p04/head2-micro-exposure-allocation-001/checkpoint_001856.pt`；SHA256 `e4772c33149410d62446180a9214b26695cb51ba1d5b7b26fac31d3ea273122b`。
- **rise_baseline_paired3762**：`baseline/paired-feedback-allocation-001/checkpoint_003762.pt`；SHA256 `e753856776bb43e44d9465b4067a0b9cd2d10fe03fc8cb88ee4177ec786f93e1`。
- **rise_baseline_to_p04_4466**：`p04/paired-head-accuracy-allocation-002/checkpoint_004466.pt`；SHA256 `f5a9a86b9acd16faeca12a9aefc63e204fff7fe90a7f328b6aad32ce7133159f`。
- **walking449**：`p04/walking-yaw-fine-allocation-002/model_448.pt`；SHA256 `8ecccd9395fade3f20f7945cd4510e655c282079cb5a367546935053c49189da`。
- **walking512**：`p04/walking-yaw-fine-allocation-002/checkpoint_000512.pt`；SHA256 `480dbcaee8666e79e541d61719cf9bcfe746444b936661784272347d603e5395`。
- **velstand256**：`p04/velstand-v8-allocation-002/checkpoint_000256.pt`；SHA256 `e18524d00d516727e2a5d5504b9423a32ee5bb72aa61d1cc86a26c526b35bcab`。
- **velstand_control320**：`p04/velstand-camera-control-allocation-001/checkpoint_000320.pt`；SHA256 `ce6fdd3834b2bf8b6e5637e07737985465043b0488cede70d6e68ad6beef2f08`。
- **velstand_camera_mild320**：`p04/velstand-camera-mild-allocation-001/checkpoint_000320.pt`；SHA256 `ab6dea2e942527549b5cf28201cc279b8e1b656e29842487575bcab607137835`。
- **velstand_head_free320**：`p04/velstand-head-free-bc-allocation-001/checkpoint_000320.pt`；SHA256 `c6f2ddcbaa39071bf243afe8690587a130d5ae55655fa0326d6e8d9c0705a9e9`。
- **velstand_head_bias384**：`p04/velstand-head-bias-allocation-001/checkpoint_000384.pt`；SHA256 `cb30a13905a1d636b26d466acc57cb1ba8e80975e20c526b175d26b20e160645`。
- **velstand_head_bias448**：`p04/velstand-head-bias-allocation-002/checkpoint_000448.pt`；SHA256 `59faeaf98b11b0361d01a796b73d044f0daafff1561ecdd0fa00c2262e533bb5`。
- **velstand_head_bias512**：`p04/velstand-head-bias-allocation-003/checkpoint_000512.pt`；SHA256 `77175a0fa9b73766e501e0647a6b55bb0cc5c936e9d1d37e03894bea4cf25e1f`。
- **velstand_head_bias576**：`p04/velstand-head-bias-allocation-004/checkpoint_000576.pt`；SHA256 `b6bcf30be62a5661fd1e448734f2b52b9de69fe0cbe1c64b188df52da520a966`。
- **sitstand640**：`p04/sitstand-history-hold320-allocation-001/checkpoint_000640.pt`；SHA256 `90b00e9867030d72bd97bf7a91490761c18306b754e2fc8aaeb80cdb3dc85ec5`。
- **sitstand_near_teacher768**：`p04/sitstand-near-teacher-lr3e6-allocation-001/checkpoint_000768.pt`；SHA256 `006bb61caee3d9565fd08a0423d39163f8acd5607987b0bd9e596f0b2a45550b`。
- **sitstand_near_teacher896**：`p04/sitstand-near-teacher-lr3e6-allocation-001/checkpoint_000896.pt`；SHA256 `a355555c4331c2847bce31d014b0089353b671348fc8ab0eba809ded568472fa`。
- **sitstand_near_teacher960**：`p04/sitstand-near-teacher-lr3e6-allocation-001/checkpoint_000960.pt`；SHA256 `e7ca35bd8f641a9dbecc9b5b14477b6a9bf09cb8b561d5d3ad5be526fe7975f5`。
- **groundpick_history512**：`p04/groundpick-health-history-allocation-001/checkpoint_000512.pt`；SHA256 `33b7cce82b2cbaad87e2c55b64fcc3fe9872ff7c43365f063a8c0045019092fb`。
- **groundpick_precision704**：`p04/groundpick-target-precision-allocation-001/checkpoint_000704.pt`；SHA256 `cc90338395cadc86f88fea7992a3a29474b961c734d349e80b4a3eec0955e64f`。
- **groundpick_precision768**：`p04/groundpick-target-precision-allocation-001/checkpoint_000768.pt`；SHA256 `52cc6d28d62051fd03fa3074fa71c98cfcb2ec6a3f01b6fdad008fb2246976c6`。
- **ballkick_health256**：`p04/ballkick-health-history-allocation-001/checkpoint_000256.pt`；SHA256 `ae17d7b3e8adf15f072f38195741d14215d4ced7f549ccae06da56fac2dea61d`。
- **ballkick_health384**：`p04/ballkick-health-history-allocation-001/checkpoint_000384.pt`；SHA256 `e3eff0691642d91a0226f0b56e65f93a8dd553d3483419bd79046ee6bcbaed9d`。
- **ballkick_precision384**：`p04/ballkick-target-precision-allocation-001/checkpoint_000384.pt`；SHA256 `dcd6773174ce3d2ae58f12aba963f4797cc0fc886b633154719e944648b50a27`。
- **ballkick_precision512**：`p04/ballkick-target-precision-allocation-001/checkpoint_000512.pt`；SHA256 `37a517218e751d963cc62f58c9617996ce02a34c164dd57713a98082e529ca82`。
- **roulade1536**：`p04/roulade-history-formal-001/checkpoint_001536.pt`；SHA256 `21b07a7e07ea721be59fad61d6efe30d9102536d9af3eaf53b15e21546ca6b8d`。

## 必须保留的评测边界

- Rise的零头命令恢复，不代表头响应/画面稳定全部通过；微幅与组合头控仍有明显失败。
- Walking449实际文件为model_448.pt，以checkpoint completed_updates449为准。
- VelStand所有候选都保留相机200Hz/20ms曝光、速度/转向和主动头误差；不以存活替代动作质量。
- SitStand使用修复命令时序后的12s协议；768 nominal外置报告失败已从完整轨迹恢复，没有重跑首批或修改阈值。
- Ground的512在新精度阶段使用共同seed参考，原报告保留；3周期来自同一256世界，不能当独立768例。
- Ball原complete与含中性头的strict complete分列；旧256是健康全历史配方节点，不是原有self-latch缺陷的170。
- Roulade中途预置latch/frontier不代表完成完整翻滚；低位折叠的低相机角速度不计健康头稳。
- 基线Ground B0只有CPU资格尝试，无训练模型。两个CPU报告错误保留且修正，不把尚未GPU准入的配置写成已通过。

主要机器可读入口：`models-and-evaluations.json`。
