"""Actual browser autoplay, manual takeover, repeatable sequence and deadman checks."""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright


def state(page):
    return page.evaluate("fetch('/api/frame').then(r=>r.json())")


def until(page, predicate, timeout=6):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        result=state(page)
        if predicate(result):
            return result
        page.wait_for_timeout(80)
    raise AssertionError(state(page))


with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    page=browser.new_page(viewport={'width':1440,'height':1200})
    errors=[]; page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8080/',wait_until='domcontentloaded')
    page.wait_for_selector('.connection.online')
    page.wait_for_function("!document.querySelector('.loading')")
    page.get_by_role('button',name='站立 R',exact=True).click()
    until(page,lambda s:s['time']<1)
    page.get_by_label('探索动作时长').fill('2')
    page.get_by_role('button',name='开始自由探索').click()
    until(page,lambda s:s['exploration']['active'])
    observed={}; commands=False; reset_poses=set()
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        current=state(page); exploration=current['exploration']
        assert not current['error'] and not exploration['failed'],current
        if exploration['action']:
            observed[exploration['number']]=exploration['action']
        commands |= any(current['command'])
        if current['spawn']!='standing': reset_poses.add(current['spawn'])
        if exploration['number']>=8: break
        page.wait_for_timeout(80)
    assert len(observed)>=8 and commands and reset_poses,observed
    page.screenshot(path='simulator/logs/exploration-desktop.png',full_page=True)
    page.get_by_role('button',name='停止自由探索').click()
    until(page,lambda s:not s['exploration']['active'] and s['command']==[0,0,0])
    # Same seed replays the first scheduled action after an explicit standing reset.
    page.get_by_role('button',name='站立 R',exact=True).click()
    until(page,lambda s:s['time']<1)
    page.get_by_role('button',name='开始自由探索').click()
    restarted=until(page,lambda s:s['exploration']['active'] and s['exploration']['number']==1)
    assert restarted['exploration']['action']==observed[1]
    page.keyboard.down('w')
    until(page,lambda s:not s['exploration']['active'] and s['command'][0]>.09)
    page.keyboard.up('w')
    until(page,lambda s:s['command']==[0,0,0])
    page.get_by_role('button',name='开始自由探索').click()
    until(page,lambda s:s['exploration']['active'])
    page.evaluate("window.dispatchEvent(new Event('blur'))")
    until(page,lambda s:not s['exploration']['active'] and s['command']==[0,0,0])
    page.wait_for_timeout(600)
    assert not state(page)['exploration']['active']
    page.get_by_role('button',name='开始自由探索').click()
    until(page,lambda s:s['exploration']['active'])
    page.get_by_role('button',name='加载行走 checkpoint',exact=True).click()
    until(page,lambda s:not s['exploration']['active'] and s['time']<1)
    page.get_by_role('button',name='开始自由探索').click()
    until(page,lambda s:s['exploration']['active'])
    page.route('**/api/control', lambda route:route.abort())
    until(page,lambda s:not s['exploration']['active'] and s['command']==[0,0,0])
    page.unroute('**/api/control')
    page.wait_for_timeout(600)
    assert not state(page)['exploration']['active']
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_timeout(300)
    assert page.evaluate('document.documentElement.scrollWidth')<=390
    page.screenshot(path='simulator/logs/exploration-mobile.png',full_page=True)
    assert not errors,errors
    result=dict(passed=True,actions=observed,reset_poses=sorted(reset_poses),keyboard_takeover=True,
                seed_replay=True,blur_stop=True,connection_stop=True,checkpoint_stops=True,page_errors=errors)
    Path('simulator/logs/exploration_browser_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    browser.close()
