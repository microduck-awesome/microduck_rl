# SitStand 外置教师可行性诊断（新增PPO为0）

旧SitStand三个配方384更新/150,994,944正式controls已关闭未合格，Lesson4原12s nominal/measured四组×三phase尾保持与同时头保持全0。此处只追加根代理授权的有限教师物理诊断，不改变冻结训练源、原奖励或预算。

## 合同与已核证据

- 冻结1856起身teacher和Lesson4的actor都是61维，14个同序绝对关节目标、HOME0、scale1；0:48真实S0/O0/IMU/上次动作读数及head命令的配置一致。23个测量/actuator关键源码和完整CAD XML SHA一致。
- SitStand slot48是0站/1坐，49:51为0；起身teacher必须接收克隆raw obs，将48:51置0。SitStand55:61为零padding，teacher相应body命令也置0；其余真实测量不改，用teacher自身冻结normalizer，绝不把起身的零twist normalizer装给SitStand学生。
- 原SitStandCommand alpha在实际reset高度初始化，然后以2s向flag变化；actor只看到flag，没有额外alpha输入。显式flag使站/坐目标可分，但不能说所有隐藏历史都消失。起身教师也不保证与原慢速alpha奖励一致，需记录实际速度/alpha而不照搬失败的线性动作插值。
- SitStand实际solver30/50、ncon64与起身10/20、ncon35不同；同观测动作合同并不代替真实目标环境迁移验证。5ms物理/20ms控制、S0/O0无去尖峰和原v5/CAD相同。
- ClearSeat固定腿：双hip−.7/knee−.8/ankle−.15，左右roll±.0873，头HOME。旧32世界3s物理证据settled .5–3s无self，末1s高.075999m、低速度、tilt最大6.45°、fault0；当时未保存feet/headfound，不能外推新增完整接触指标。旧128/128线性过渡失败使用的是原有自干涉SIT目标，也不证明所有ClearSeat过渡不可行。

## 唯一GPU1过渡probe

`probe_sitstand_external_teachers_001.py`，64世界（原四initial-state/first-command组合各16），12秒：4s首次指令/4s反向/4s返回，head全HOME。站分支为冻结1856均值，坐分支为ClearSeat固定14目标，均走真实P04action/feedback；不做关节线性轨迹。名义S0/O0、原startup DR，禁独立评测push；完整原奖励照常只调用一次。

`held_compute`在native前固定flag/head及极长resample timer，然后原生compute负责fresh-height alpha与slew；每步动作前重算实际obs(update_history=False)，teacher适配只改命令槽。保存rawobs/teacherobs/执行target/count/完整qpos/qvel和原stable；新增独立bothfeet/headfound/self/contactforce与严格body/headHOME保持，200Hz相机/20ms曝光按过渡与末端健康子集分开。

当前PID191292/session8235；输出 `sitstand-external-teachers-probe-002`。本完整批次38,400 diagnostic controls，正式新增0；001首次反向断言停止前已执行12,800诊断controls。首次启动导入期静态审查捕获async assert传Pythonbool，0物理停止后改Tensor.all；原脚本/log/修复receipt保留，实际只有一批物理probe。

CPU合同：`sitstand-teacher-contract-001.json`。该结果必须标外置教师诊断，不算原384单策略成功，也不能直接签BC或新PPO资格。根代理GPU0另做实际R1固定坐姿9个头小阶跃，结果独立，不混为本probe。

## 指令缓存时序修复（原件保留）

001首次反向前新增断言抓到实际flag与raw actor命令不符；锁定ObservationManager.compute(False)在cache存在时直接返回旧obs，复用的旧SitStand evaluator“改指令再compute(False)”并不刷新命令，切换首帧有20ms旧命令。该判断来自原生源码304–311与实际GPU断言；不是反馈或GPU硬件故障。旧384评测原件保留并注明局限，根代理另做两条件256×12s修正重评；尚不把这个单帧问题归因为全部失败。

002撤销额外compute：每个动作只用已有native obs，assert其flag等于当前phase目标；预设下一控制desired后env.step，native末端command compute自然更新flag/alpha并生成下一obs。既不清cache/延迟，也不重采物理感知或注入真值。600步CPU调度边界检查已过，实际GPU断言仍保留；源码`probe_sitstand_external_teachers_002.py:28/67/75`。整个旧001执行脚本、日志、failure.json和12,800预算保持。

## 完整002结果：只有局部端点证据，不进入PPO

002完成600控制/64world/38,400诊断步，两次反向及动作前真实obs/command断言全部通过；源和冻结1856 actor/normalizer不变，feedback/nonfinite全0。001失败12,800步不抹除，总诊断51,200；正式新增0，旧SitStand150,994,944仍不变。GPU1释放，无active PID。

- 初始站→站：原body/严格body/头HOME均16/16，整个4s无head/self接触，最大tilt3.52°。
- 初始坐→坐：原body16/16，补双足等后的严格body/头HOME15/16（缺1足接触），无head/self，最大tilt5.16°。
- 每次站→坐固定ClearSeat：终态0/16，全部明显倾倒/头与自接触；典型最大tilt约87–92°，头力峰约35–41N。不能把静态SIT标签覆盖整个坐下路径。
- ClearSeat坐→1856站：尾部16/16身体+头HOME，但全部tilt>90°，最大tilt中位约93.06°、身体最低高度中位45.37mm/最小42.70mm、头力峰58.85N，约1.18s才首次严格站稳。第二轮坐→站同类结果。官方允许过渡头辅助，单凭head触地不能判失败；这里同时保存大倾倒/下降/冲击事实，不能写为直接平顺起立。

每阶段原stable与独立feet/head/self严格指标并列，200Hz相机/20ms曝光用各自真实健康分母；主动过渡不当静止稳像。head本批一直HOME，没有证明主动头指令准确；根代理另测R1小头阶跃，证据独立。

最终机器记录 `sitstand-teacher-probe-closeout-002.json`、probe002/report.json与transition-analysis.json。建议边界仅为目标附近的端点标签；本结果未签完整过渡teacher、BC或新PPO资格，后续单一方案由根代理决策。
