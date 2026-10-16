# BallKick 成熟目标精度固定段

当前：成熟精度256连续正式及原nominal/measured四节点评测全部完成，预算关闭，无自动续训。新增150,994,944、BallKick全历史477,757,440；最终曲线和头稳/健康分母见本文末及 `p04/ballkick-target-precision-final-review.json`。

父固定原256：两条件完整且头中性48/41，优于192的1/0、320的24/17、384的20/18。原完整身体任务51/44。父256/320/384 actual manager均action_rate−1、COM.015、headCOM.01、push.3；不能称后期课程升级导致退化，也不把冻结相同设置伪装修复。

仅正奖改为12*exp(-((forward−1)/.2)^2−(lateral/.2)^2)*healthyvalid，沿原resetheading投影。成熟有效触球已两条件256/256，故只续成熟父，不用于fresh发现。原overshoot−4、direction−4、quiet−2、头HOME/姿态全部保留。actor61仍无直接球位置/速度，critic82有；这是现有信息下的统计精度优化，不保证能闭环纠正所有不可观测球位扰动。Gaussian等高线不是原持续1s矩形验收gate，原5s完整/回稳/中性头/相机评测不变。

固定N24576/T24/E5/M24×256=150,994,944新controls/30,720Adam，从原256到512，clock6144→12288。CPU公式/导数/配置差分/资产与父选择→同源realterminal→成熟64×5/ONNX→actual24576×16，通过才正式。全部权重/N/Adam/LR/RNG精确继承原256，不用验证权重；没有自动第二段。

每64存320/384/448/512，正式连续结束后原nominal/measured统一复评。生产patch、完整源SHA和rootdecision见 `ballkick-target-precision-plan-001.json`、`ballkick-target-precision-production.patch`、`ballkick-precision-stage-root-decision-001.json`。

实际资格：成熟64×5精确父actor/critic/N/Adam、LR.0002562890625/clock6144，32timeout、ONNX2.98e−7；actual24576×16为9,437,184验证controls、36,830timeout，contact216,697/1,572,864，五项oracle≤9.54e−7，feedback/nonfinite0，max meanKL.01255/clip.15709；稳态median89,687.20/aggregate88,035.61 controls/s。正式重新从原256加载，不继承validation272；原完整验收在320/384/448/512后统一执行。

## 成熟精度256续段最终曲线（同原门槛）

| 节点 | 条件 | raw足触球 | 健康valid | clean | 滚速方向 | 身体保持 | 身体+头中性 | 完整 | fault |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 320 | nominal | 256 | 256 | 255 | 143 | 108 | 76 | 60 | 0 |
| 320 | measured | 256 | 256 | 256 | 126 | 104 | 80 | 50 | 0 |
| 384 | nominal | 256 | 256 | 249 | 128 | 119 | 67 | 67 | 0 |
| 384 | measured | 256 | 256 | 256 | 105 | 122 | 71 | 51 | 0 |
| 448 | nominal | 256 | 254 | 252 | 123 | 94 | 32 | 44 | 0 |
| 448 | measured | 256 | 255 | 253 | 99 | 96 | 28 | 35 | 0 |
| 512 | nominal | 256 | 255 | 254 | 109 | 84 | 21 | 28 | 0 |
| 512 | measured | 256 | 251 | 250 | 99 | 80 | 18 | 19 | 0 |

新增正式150,994,944，累计477,757,440；实际30,720次Adam、nativeclock12288。完整formal共享进程阶段1875.297s/80517.89controls/s；稳态median81001.48、aggregate81318.11。完整worker含资格和8次末评另列2246.407s，不混为学习吞吐。

各条件320/384/448/512的原始qpos/qvel/actor_obs/origins配对核验、200Hz首次触球/自碰时刻和分窗口相机RMS/20ms曝光及健康分母见 `p04/ballkick-target-precision-final-review.json`。完整资格不由某一分项代替；未完成原全门槛者仍未合格，runtime_qualified/hardware_tested=false。预算到此关闭，不自动续训。
