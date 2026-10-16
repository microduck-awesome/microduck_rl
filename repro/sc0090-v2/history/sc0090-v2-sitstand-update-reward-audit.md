# SitStand PPO更新与奖励曝光审计

当前：正确TransitionRelease配方fresh128已结束且未合格；SitStand正式累计226,492,416，原全部失败预算/检查点保留。之后唯一M16验证16更新已完成，正式0；GPU1已释放，未自动启动新正式段。

## 可核对的优化历程

| 方案 | N×T / rollout | 每mini-batch | epochs×M / rollout | 更新数 | 实际或约定Adam步 |
|---|---:|---:|---:|---:|---:|
| 本次fresh正式M4 | 16384×24=393216 | 98304 | 5×4=20 | 128 | 2560，日志及checkpoint原生step核实 |
| M4实际N验证 | 393216 | 98304 | 20 | 16 | 320，原生Adam step |
| M16实际N验证 | 393216 | 24576 | 80 | 16 | 1280，实际step回调+原生Adam双证据 |
| 等50,331,648控制的N4096算术对照 | 4096×24=98304 | 24576 | 20 | 512 | 10240，仅算术、未运行 |
| 历史基础jngaedcs@1250 | 文档N4096 / T24 | 24576 | 20 | 文档节点约1250 | 约25000，未取得旧raw checkpoint精确步数 |

历史来源为固定official worktree `docs/sitstand_policy.md`（基础c832147、硬化77ffadf），已从对应git对象取真实cfg，T24/E5/M4一致；本地未找到jngaedcs/vfhxgrum/r78hyhck原始W&B configs/checkpoints，保留文档证据级别及旧label±1更新歧义。历史XL330物理与12s episode不同，不能据这张表宣称V2算法错误或学习效果等价。

本次fresh clock0→3072，旧448 clock10752；全部10项课程都尚在stage0，最早边界12000（500更新）。等交互N4096×512则clock12288，会跨第一边界，不能机械当完全同课程对照。M16只匹配历史单minibatch大小，当前rollout仍4倍，未匹配历史策略刷新频率。

## 命令与真实健康数据

fresh128共50,331,648控制，命令near-complete（|flag−alpha|<.02）37,425,754=74.36%，sit17,494,599、stand19,931,155；descending7,556,496、rising5,349,398。健康控制点只有176，非176个成功episode。

训练没有单独保存exact g=1计数，不能将near-complete代理冒称exact。实际保存评测中nominal/measured g=1各90050/90054 of153600，所有四组×三phase末500ms全部g=1，健康保持仍0。因此末端失败不是该门控仍关闭。站初态的STAND末端height/leg gate全过但tilt15 gate全失败；同初态SIT命令height和leg gate都失败。坐初态倒地簇height/tilt gate均失败，leg gate却可全过。

## 奖励分项与局部敏感度

同已冻结公式，正确固定HOME/SIT的9项可重建分项总和约13.691/12.196（健康参考32/31worlds，真实R1最后1s）。正确composite为2.974/2.958（上限3）。不是同动作反事实，也不是含扭矩/接触成本的完整return。

fresh128实际SIT倒地末态仍拿pose约3.78/4、宽height约.896、细height约.46、upright约.93，composite却只有.019/3。当前站姿末端neck误差114.27°，其它头轴5.21/2.74/4.32°；head Gaussian四轴均值仍约.547/.75，但neck本轴奖励贡献仅2.60e−8、对物理neck误差的奖励斜率约4.11e−7/rad。composite受head乘项抑制约.00290。这证明当前远端neck奖励敏感度很低，不等于PPO/action梯度，也未证明全部失败的因果根源。

真实训练TensorBoard全部21项已存：末完整episode平均pose2.640、head.512、height宽/细.752/.456、composite.102、stillness.0228、action_rate−.913、neck_strain−.447。它们来自随机训练混合及完成episode，不能冒称最终确定性策略某静态簇的全部回报。保存评测仅能严格重建9个无状态分项；完整3D线速度/扭矩/角动量/接触历史不足，不捏造完整return排序。根qpos/速度与派生几何存在5ms边界差，重建使用已保存同reward边界height/tilt，不用错位qpos替代。

## 唯一M16诊断结果

001外置计数器在attach前包algorithm.update，被原upstream身份护栏拒绝：0采样、0update、0optimizer.step，原件保留，不能计作M16数值不稳。002保留身份护栏，仅包原生optimizer.step并在logger边界核每update80；数值源未改、agent唯一M4→16。两臂初始actor/critic/N/Adam/LR/CPU+活动CUDA RNG逐位相同；旧M4未存完整物理初态，未称全部状态配对。

| 16更新比较 | M4 | M16 |
|---|---:|---:|
| 控制数 / sample-visits | 6,291,456 / 31,457,280 | 相同 |
| 原生Adam步 | 320 | 1280 |
| 稳态collection中位 | 3.419s | 3.470s |
| 稳态learning中位 | .388s | .711s |
| 排除首3更新后13次wall | 51.525s | 57.104s（+10.8%） |
| 稳态吞吐median / aggregate | 100961 / 99211 | 92078 / 89518 |
| max update KL / maxclip | .01648 / .22934 | .01530 / .23189 |
| 健康控制点（各629万） | 11 | 19 |
| height误差 / tilt均值 | .03780m / 1.2845rad | .03812m / 1.2752rad |
| 末完整episode composite | .12038 | .12068 |

两臂fault/nonfinite0，均16384原生timeout、容量/oracle通过。M16数值可用，4倍Adam步约增加10.8%稳态耗时；目前没有实质技能收益证据，不能将11→19极少控制点写成功率增益。M16增加优化步，不增加epochs或总sample-visits。它只验证优化密度因素；远端neck奖励饱和是另一项独立证据，不同时改奖励。无新增正式授权，不自动下一段。

完整机器证据：`sitstand-fresh-key-cpu-audit-001.json`、`sitstand-fresh-reward-and-update-audit-001.json`、`sitstand-fresh-m4-m16-comparison-001.json`，外置脚本/SHA/实际agent/source配置均归档。
