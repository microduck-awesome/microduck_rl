# 本地交互演示验证记录

日期：2026-09-15。环境：Linux x86_64、Python 3.12、MuJoCo 3.10、CPU ONNX Runtime、
Node.js 22.22、Chromium（无头 SwiftShader）。浏览器渲染、物理执行和 checkpoint 导出分别独立。

## 默认模型

| 任务 | checkpoint | ONNX SHA-256 |
| --- | --- | --- |
| 行走 V2 | 本轮 `model_5999.pt` | `20919458db8da2fcaed81f72c4eeddd7e34c62a52e1364b659cf906b35cfe2d1` |
| 起身 V3 | 本轮 `model_5999.pt` | `c438797d3df768b02bc086bd518902ca7d07ad9020b2fd5dde12a681e55116a5` |

完整来源、任务和电机摘要在 `models/manifest.json`。本轮编号不等于累计训练轮数。

## 自动检查

- Python：**40 passed**，覆盖输入序号、控制权交接、断连与过期事件、并发提交、非法数值、
  BAM 状态重置、80 rpm 参数、低速策略选择、模型加载校验失败保留原状态、各起身姿态、
  checkpoint 任务分离、源文件变化、无训练日志的克隆、CPU 导出进程隔离及超时退出。
  自由探索检查覆盖随机序列复现、整轮动作覆盖、指令边界、物理步长对应的完整动作时长、
  暂停不推进计时、等待起身、失败锁定、停止后不重播、断连退出及真实策略的完整 16 种动作序列。
- 前端：**10 passed**，包含键盘映射、低速和组合指令、原始关节角与四元数绘制。
- `npm run build`：通过。
- 数值对照：最终行走/起身、旧行走 `4050`、中途起身 `5300` 共 **4 份不同模型**，
  每份比较 97 组输入；使用训练库 `MLPModel` 严格加载原 checkpoint，包括观测归一化。
  与 CPU ONNX 输出的最大绝对误差均为 **4.7684e-7**，通过 `rtol=atol=2e-5`。
  输入含全零，以及按训练观测均值、标准差生成的三个尺度样本。

## 真实浏览器操作

Chromium 检查通过：三维资源加载、键盘前进和转向、松键和失焦停止指令、0.03 m/s 指令
保持行走模式、暂停/继续、策略切换、五种倒地姿态起身、桌面和手机尺寸无横向溢出。
页面没有 JavaScript 异常。

最终起身模型在固定初始姿态的一次浏览器检查中，达到标准并连续稳定 0.5 秒的仿真用时：

| 初始姿态 | 用时 |
| --- | --- |
| 坐姿 | 0.88 s |
| 俯卧 | 1.22 s |
| 仰卧 | 1.26 s |
| 左侧卧 | 1.18 s |
| 右侧卧 | 1.30 s |

这些是固定标称参数的单次展示结果，不代表随机化成功率或实机起身时间。
低速检查确认指令和模式正确，不代表实际速度已经准确跟踪。

网页 checkpoint 选择检查通过：分别加载旧行走 `4050` 和中途起身 `5300`，切回默认模型，
再加载缓存并再次恢复默认。加载后模型身份、页面文件名与站立重置均符合预期。
本机首次 CPU 导出分别约 **18.8 s / 18.6 s**；缓存加载约 **0.56 s / 0.80 s**。
时间随主机负载和编译缓存变化，不是固定性能承诺。

自由探索浏览器检查通过：随机连续完成转弯、停步、仰卧/俯卧起身、后退、右侧卧/坐姿起身，
再进入绕弯；相同序列编号复现首段动作。停止按钮、方向键接管、失焦、控制连接中断及
加载 checkpoint 均退出探索，不会在恢复连接后自行重启。桌面/手机布局无横向溢出，
无 JavaScript 异常。原手动键盘及五种起身浏览器检查也再次通过。

移除倒地选项后，相关 **33 项 Python 检查**、前端构建与自由探索浏览器检查再次通过。
每轮固定混合行走与全部五种起身姿态，旧页面发送的关闭倒地选项也不会禁用这些动作。

## 复现

```bash
# 仓库根目录
uv run --no-sync --with pytest pytest simulator/tests tests/test_infer_policy_bam.py -q
cd simulator/app && npm test && npm run build
cd ../..
./scripts/play_sc0090_local.sh
```

服务运行时，在另一终端执行以下检查。首次浏览器检查需要安装 Chromium；后两个检查
需要本地保留相应训练 checkpoint。

```bash
uv run --no-project --with playwright playwright install chromium
uv run --no-project --with playwright python simulator/tests/browser_check.py
uv run --no-project --with playwright python simulator/tests/checkpoint_browser_check.py
uv run --no-project --with playwright python simulator/tests/exploration_browser_check.py
.venv/bin/python simulator/tests/policy_parity_check.py --include-cache
```

浏览器检查会操作共享演示进程，运行时请关闭其他控制窗口。结果 JSON 与截图写入
`simulator/logs/`，导出日志和缓存位于 `simulator/models/cache/`；都不加入 Git。

验证不证明软件绝无竞态或数值问题。运行时采用单线程物理/推理、有界输入队列、
独立导出进程、切换前校验及数值异常可见暂停；CPU 演示与 GPU 训练的轨迹不要求逐位一致。
