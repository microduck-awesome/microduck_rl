"""Verify static Pages video playback, metadata, navigation and responsive layout."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:8000/')
args=parser.parse_args()
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]; requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:requests.append(r.url))
    page.goto(args.url,wait_until='domcontentloaded')
    assert '5999' in page.locator('.intro').inner_text()
    assert page.locator('video').count()==17
    assert page.get_by_text('本轮训练已完成',exact=True).is_visible()
    page.locator('video').evaluate_all("vs=>vs.forEach(v=>{v.preload='metadata';v.load()})")
    page.wait_for_function("[...document.querySelectorAll('video')].every(v=>v.readyState>=1)",timeout=60000)
    clips=page.locator('video').evaluate_all("vs=>vs.map(v=>({file:v.getAttribute('src'),duration:v.duration,width:v.videoWidth,height:v.videoHeight}))")
    assert sorted(v['duration'] for v in clips)==sorted([60]+[4]*5+[8]*11)
    assert all((v['width'],v['height'])==(640,480) for v in clips)
    page.get_by_label('播放速度',exact=True).select_option('0.5')
    assert page.locator('video').evaluate_all('vs=>vs.every(v=>v.playbackRate===0.5)')
    page.locator('#exploration video').evaluate('async v=>{v.muted=true;await v.play()}')
    page.wait_for_function("document.querySelector('#exploration video').currentTime>0.1")
    page.get_by_role('button',name='全部暂停').click()
    assert page.locator('video').evaluate_all('vs=>vs.every(v=>v.paused)')
    page.get_by_label('播放速度',exact=True).select_option('1')
    page.screenshot(path='simulator/logs/pages-desktop.png')
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth')<=390
    page.screenshot(path='simulator/logs/pages-mobile.png')
    page.locator('nav').get_by_role('link',name='历史基线 ↗').click()
    assert page.locator('video').count()==20 and '历史基线' in page.title()
    assert not errors,errors
    assert all(urlsplit(url).netloc==urlsplit(args.url).netloc for url in requests),requests
    assert not any('/api/' in url or '127.0.0.1:8080' in url for url in requests)
    report=dict(url=args.url,passed=True,videos=clips,playback=True,rate=True,pause=True,
                mobile=True,baseline_videos=20,external_requests=False,page_errors=errors)
    Path('simulator/logs/pages_browser_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    browser.close()
