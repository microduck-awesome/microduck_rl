#!/usr/bin/env python3
"""Build the static Pages video index from the recorded model/video manifest."""
from html import escape
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
data=json.loads((ROOT/'docs/videos/latest/gallery.json').read_text())
videos=data['videos']
explore=next(v for v in videos if v['kind']=='exploration')
models=data['models']['policies']
labels={kind: f'{entry["task"].rsplit("-",1)[-1]} / {Path(entry["checkpoint"]).stem.removeprefix("model_")}'
        for kind,entry in models.items()}
recorded_date=datetime.fromisoformat(data['recorded_at']).astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()
assessment=json.loads((ROOT/'docs/videos/latest/assessment.json').read_text())
for kind,entry in models.items():
    if assessment[kind]['checkpoint_sha256'] != entry['checkpoint_sha256']:
        raise ValueError(f'{kind} assessment does not describe the recorded checkpoint')
status=assessment['display_status']


def video(v, featured=False):
    return (f'<video controls playsinline preload="none" poster="videos/latest/{escape(v["poster"])}" '
            f'src="videos/latest/{escape(v["file"])}" aria-label="{escape(v["title"])}"></video>')


def card(v):
    if v['kind']=='recovery':
        result=(f'首次站稳 {v["rise_time_s"]:.2f} 秒 · 含 0.5 秒稳定确认'
                if v['rise_time_s'] is not None else '本段未完成站稳，保留完整尝试')
    else:
        vx,vy=v['mean_velocity_m_s']
        result=f'实测前后 {vx:+.3f} / 侧向 {vy:+.3f} m/s'
        result+=f' · 转向 {v["mean_yaw_rate_rad_s"]:+.3f} rad/s'
        if v['fell']: result+=' · 期间触发起身'
        if v['id']=='walk_forward_003' and abs(vx)<.005:
            result+=' · 本次低速指令下基本未移动'
        if 'command_schedule' in v:
            result+=' · 2 秒切换指令，取第 3–4 秒平均'
    return (f'<article data-kind="{v["kind"]}"><div class="card-title"><h3>{escape(v["title"])}</h3>'
            f'<span>{v["duration_s"]:g} s</span></div>{video(v)}<p>{escape(result)}</p>'
            f'<a class="download" href="videos/latest/{escape(v["file"])}" download>下载 MP4 ↗</a></article>')


walk=''.join(card(v) for v in videos if v['kind']=='walking')
recovery=''.join(card(v) for v in videos if v['kind']=='recovery')
html=f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Microduck SC0090 行走 {escape(labels['walking'])}、起身 {escape(labels['recovery'])} 与自由探索视频。修正物理实现后重新录制的完整原速仿真回放。">
<title>SC0090 最新训练视频 · 行走、起身与自由探索</title><link rel="stylesheet" href="gallery.css"></head>
<body><main>
<header><div><span class="eyebrow">MICRODUCK / SC0090</span><h1>行走、倒地，再站起来。</h1>
<p class="intro">最新训练视频 · 行走 <b>{escape(labels['walking'])}</b> · 起身 <b>{escape(labels['recovery'])}</b> · {recorded_date} 更新</p></div>
<span class="badge">物理修正后重新录制</span></header>
<p class="muted">{escape(status)}</p>
<nav><a href="#exploration">自由探索</a><a href="#recovery">五种姿态起身</a><a href="#walking">行走与转向</a><a href="legacy-5999.html">修正前 5999 视频 ↗</a><a href="baseline.html">历史基线 ↗</a></nav>
<section id="exploration" class="featured">
<div class="feature-copy"><span class="eyebrow">CONTINUOUS REPLAY / 1×</span><h2>自由探索</h2>
<p>自动随机组合行走、转弯、停步、扰动和五种倒地姿态，模型起身稳定后继续探索。</p>
<p class="muted">这是训练机录制的完整 {explore['duration_s']:g} 秒视频。倒地测试会明确显示“重置为某姿态”，之后由模型自行起身。</p>
<div class="chips"><span>12 V · 80 rpm</span><span>序列编号 1</span><span>每段指令 4 秒</span></div>
<a class="download" href="videos/latest/{explore['file']}" download>下载完整探索视频 ↗</a>
</div>{video(explore,True)}</section>
<div class="playback"><span>所有视频按仿真时间原速录制</span><label>播放速度 <select id="rate" aria-label="播放速度">
<option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1× 原速</option><option value="2">2×</option>
</select></label><button id="pause">全部暂停</button></div>
<section id="recovery"><div class="section-title"><div><span class="eyebrow">RECOVERY {escape(labels['recovery'])}</span><h2>五种姿态，逐一看起身。</h2></div>
<p>每段完整 4 秒 · 每种姿态固定录制一次<br>站稳时间包含连续 0.5 秒稳定确认</p></div><div class="grid">{recovery}</div></section>
<section id="walking"><div class="section-title"><div><span class="eyebrow">WALKING {escape(labels['walking'])}</span><h2>从低速，到侧移和转弯。</h2></div>
<p>每段完整 8 秒 · 指令速度与实测速率分别标注<br>恒定指令取第 2–8 秒平均；启停取切换后第 3–4 秒</p></div><div class="grid">{walk}</div></section>
<section class="conditions"><h2>这些视频记录了什么</h2>
<p>使用与本地模拟器相同的 CPU MuJoCo、SC0090 BAM 电机模型与归一化 ONNX 策略，策略运行于 50 Hz。
每种场景固定录制一次，没有按成功结果挑选；使用标称物理参数，无训练随机化。画面为仿真，单次结果不代表随机化成功率或实机表现。</p>
<p>起身段未做自动重置；自由探索中的姿态重置和扰动有明确字幕与时间记录。0.03 m/s 保留为低速诊断，当前训练的纯行走最低目标为 0.08 m/s；这不是舵机实测最低速度。</p>
<p>本次使用物理修订 {data['dynamics_revision']}，已修正摩擦索引、摩擦公式和观测时序。修正前的 5999 视频保留为历史记录，物理实现与本次不同，不能作为严格的前后效果对照。</p>
<p>两项均从各自的 5999 checkpoint 保留训练状态续训 401 次更新至 6400；完成的是本轮训练预算，当前验收结论见页首。编号不代表从零开始的累计训练量。</p>
<a href="videos/latest/gallery.json">本次视频、模型摘要与测量数据 JSON ↗</a><a href="videos/latest/assessment.json">本轮评估记录 JSON ↗</a><a href="legacy-5999.html">查看修正前视频 ↗</a></section>
<footer><span>Microduck · SC0090 12 V / 80 rpm</span><a href="https://github.com/microduck-awesome/microduck_rl">GitHub 仓库 ↗</a></footer>
</main><script src="gallery.js"></script></body></html>'''
(ROOT/'docs/index.html').write_text(html)
print(ROOT/'docs/index.html')
