# GroundPick B0 基线专职台账

当前：B0 生产源已固定，worker001/002均已退出，累计0仿真、0PPO。002全部CPU配置/资产/agent断言通过后，在报告写盘遇到slice序列化错误；只修外置canonical编码，保留原失败，尚未启动terminal/GPU资格。按用户新优先级先整理repro分支并推送，根代理负责后续驻留资格接续；本代理未自行重启。

## 目标和固定方案

比较基线动作发现、后续有效 DR/量化适应及完整 P04 续训的总时间和共同验收结果。当前仅 B0；后续阶段尚未启动。

B0 保持 GroundPick History 的 V2 CAD/full-contact/v5、HOME0、61 actor/74 critic/14 action、25% 真实低位 reset、4s phase/20s episode、奖励与课程。只切换为官方 native JointPositionAction、q 噪声 ±.001rad、v 噪声 ±.25rad/s、v 延迟1控制步、原 encoder bias。原生 action 本身减 encoder bias，actor q 同时加 bias；不是隐藏的理想观测。mouth 独立目标0。无 P04 stream/count/residual。critic 保持物理 privileged 观测，真实无噪声和无延迟须通过资格。

固定 fresh512，N24576/T24/E5/M24，301,989,888 控制、61,440 PPO Adam 上限。使用原 P04 untrained initialization 的共同随机 actor/critic（61/74/14维、clock0、N count0、Adam空），不是已训练策略。验证权重不继承。GPU1 统一驻留 worker 由根代理控制。

## 本代理工作与验证

仅修改 ground_pick_baseline.py、任务注册、ground_pick_metrics.py；根代理负责 trainer、worker、共同 runtime evaluator。指标内部物理条件采用 ready 恒真作为不适用的单位因子，公开 feedback_not_ready_fraction=null，feedback_fault_metric_applicable=false；绝不伪装 P04 反馈已通过。原 P04 分支仍要求真实反馈对象。

CPU 检查逐项比对真实官方 Ground 配置，回滚声明的 action/观测/encoder/mouth/标记差异后与 P04 环境完全一致；v5/CAD/runtime/terminal 不修改。worker001 失败来自额外验证断言检查 raw critic noise=None；native ObservationManager 实际在 enable_corruption=False 时清除噪声，delay 独立控制。已修正噪声判据，delay0 硬检查不删。

真实 terminal 入口验证 32 环境、低位 phase/reset 配对、3 控制步真实 action 与 encoder-bias 关系、mouth0、7 项奖励/载荷 oracle、200Hz 接触采样；12 个 phase/force_full successor 情形检查 live 张量/RNG 不变、物理载荷力矩、partial reset。此处接触采样不作端点健康结论。

详细源 SHA 和入口：`baseline/groundpick-baseline-implementation-001.json`。报告预定 `groundpick-baseline-cpu-001.json`、`groundpick-baseline-terminal-001.json`；应以真实结果更新状态，不把准备完成当资格 PASS。

## 复现清单交接

已生成 `ART/repro-staging/models-and-evaluations.json` 与同名MD：25个真实检查点逐SHA、各源工作树与run清单exact、训练N/T/E/M/clock/实际PPO与BC步数、原评测和限制均记录。权重不上传。每条谱系成本与全研究累计分别列出；完整谱系/项目wall不足处为null，不从稳态吞吐推算。B0在清单中仍无训练模型。
