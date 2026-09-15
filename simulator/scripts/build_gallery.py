#!/usr/bin/env python3
"""Build the static Pages video index from the recorded model/video manifest."""
from html import escape
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
data=json.loads((ROOT/'docs/videos/latest/gallery.json').read_text())
videos=data['videos']
explore=next(v for v in videos if v['kind']=='exploration')


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
    return (f'<article data-kind="{v["kind"]}"><div class="card-title"><h3>{escape(v["title"])}</h3>'
            f'<span>{v["duration_s"]:g} s</span></div>{video(v)}<p>{escape(result)}</p>'
            f'<a class="download" href="videos/latest/{escape(v["file"])}" download>下载 MP4 ↗</a></article>')


walk=''.join(card(v) for v in videos if v['kind']=='walking')
recovery=''.join(card(v) for v in videos if v['kind']=='recovery')
html=f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Microduck SC0090 最新模型的行走、五种姿态起身与自由探索视频。行走 V2 和起身 V3 checkpoint 5999，完整原速仿真回放。">
<title>SC0090 最新训练视频 · 行走、起身与自由探索</title><link rel="stylesheet" href="gallery.css"></head>
<body><main>
<header><div><span class="eyebrow">MICRODUCK / SC0090</span><h1>行走、倒地，再站起来。</h1>
<p class="intro">最新训练视频 · 行走 V2 / 起身 V3 · 两项均为 checkpoint <b>5999</b> · 2026-09-15 更新</p></div>
<span class="badge">本轮训练已完成</span></header>
<nav><a href="#exploration">自由探索</a><a href="#recovery">五种姿态起身</a><a href="#walking">行走与转向</a><a href="baseline.html">历史基线 ↗</a></nav>
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
<section id="recovery"><div class="section-title"><div><span class="eyebrow">RECOVERY V3 / 5999</span><h2>五种姿态，逐一看起身。</h2></div>
<p>每段完整 4 秒 · 每种姿态固定录制一次<br>站稳时间包含连续 0.5 秒稳定确认</p></div><div class="grid">{recovery}</div></section>
<section id="walking"><div class="section-title"><div><span class="eyebrow">WALKING V2 / 5999</span><h2>从低速，到侧移和转弯。</h2></div>
<p>每段完整 8 秒 · 指令速度与实测速率分别标注<br>下方实测值取第 2–8 秒平均</p></div><div class="grid">{walk}</div></section>
<section class="conditions"><h2>这些视频记录了什么</h2>
<p>使用与本地模拟器相同的 CPU MuJoCo、SC0090 BAM 电机模型与归一化 ONNX 策略，策略运行于 50 Hz。
每种场景固定录制一次，没有按成功结果挑选；使用标称物理参数，无训练随机化。画面为仿真，单次结果不代表随机化成功率或实机表现。</p>
<p>起身段未做自动重置；自由探索中的姿态重置和扰动有明确字幕与时间记录。旧 checkpoint 3450 / 3200 的视频和当时测量保留在历史基线页面，测试条件与本次不同。</p>
<p>本轮两项训练的目标均已完成，最终文件编号为 5999。这个编号属于本轮训练，不等于从零开始累计训练了 6000 轮。</p>
<a href="videos/latest/gallery.json">本次视频、模型摘要与测量数据 JSON ↗</a><a href="baseline.html">查看旧模型视频 ↗</a></section>
<footer><span>Microduck · SC0090 12 V / 80 rpm</span><a href="https://github.com/microduck-awesome/microduck_rl">GitHub 仓库 ↗</a></footer>
</main><script src="gallery.js"></script></body></html>'''
(ROOT/'docs/index.html').write_text(html)
print(ROOT/'docs/index.html')
