# 训练参数、课程和评测口径

每个模型的准确值以 `recipes.json` 关联的 `run.json` 和 checkpoint 为准。继续训练时 checkpoint 保存的实际 LR、Adam、normalizer 与 clock 优先于工厂里的初始参数；加载父模型不等于重置课程。

| 参数 | 本轮常用配置及作用 |
|---|---|
| N 环境数 | 起身历史8192、后期32768；walking32768；VelStand/Ground/Ball24576；Sit/Roulade16384。按显存、接触容量和学习效果实测选择，非统一8192 |
| T rollout | 24，来自本地官方配方 `AGENTS.md`/runner。50Hz下每世界0.48s采样；不等于episode长度。T变大会影响延迟、GAE覆盖和batch |
| E epochs / M minibatches | 多数E=5；起身/walking M=4，Ground/Ball M=24，Sit/Roulade按各节点记录。每update实际Adam次数E×M，不能只比updates |
| learning_rate / schedule | 新任务常见0.001/adaptive；续训保留实际LR。头部或BC尝试另有有限LR方案。增大可能加速也可能造成KL/clip过大 |
| gamma / lambda | 常见0.99/0.95：回报时域与GAE偏差/方差折中；不随N盲改 |
| clip / desired KL | 常见0.2/0.01，限制策略改变量并控制adaptive LR；监视实际KL与clip fraction |
| entropy / exploration std | 常见entropy0.01，起身后期有不同方案/头部std上限；探索太小可锁死，太大损害精细稳定 |
| value loss / grad norm | 常见1.0/1.0，clipped value loss；actor/critic同时检查数值和timeout bootstrap |
| normalization / advantage | actor、critic有独立统计；成熟父模型有冻结统计方案。advantage归一化和minibatch配置见原记录 |
| symmetry / BC | 任务相关：只对有意义的对称任务使用；teacher的训练成本、BC Adam次数及mask要单列 |

PPO batch=N×T。增大N不等价于保持学习轨迹，minibatch尺寸和每单位交互的Adam步数也会改变。吞吐用50Hz policy control transitions/s；200Hz物理子步不能乘4冒充交互。稳态、完整正式阶段、包含编译/初始化/评测的完整worker分别报告。

课程继续借鉴已验证历史：渐进spawn难度、给稀有命令明确曝光、真实零速采样、turn-in-place桶、后期再增加动作平滑税；奖励必须与任务物理指标同向。平台期先查实际 manager 值和命令曝光，再查升级门槛与精度梯度。有限预算阶段不因几次短试验就宣判失败，也不无上限追加。

训练和评测共同使用测量定义。P04=S0/O0/R1，O0不是O1；最终分支不去除合法尖峰，保留非法包/count/NaN/时间支持检查，不向actor注入物理真值。物理真值只用于明确标注的评测/诊断和privileged critic。

头部关节精度/响应时间与世界相机稳定性是不同指标；后者按200Hz、20ms曝光计算，分别报告全体与健康样本分母。走直、转向、步态、零速跺脚必须分别验收。保留接触和无效反馈的失败样本，不能通过删失败轨迹提高通过率。

Ground最新CPU/GPU相机统计有8640项对照，1项绝对误差2.19345e-5rad/s，审计采用atol2e-5+rtol1e-6；这只是浮点复核容差，未放松技能的物理验收门槛。Sit首次评测报告的float/bool错误已修复并利用原轨迹恢复；未把报告失败算作新训练。
