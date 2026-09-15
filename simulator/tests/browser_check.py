import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright

def state(page):
 return page.evaluate("fetch('/api/frame').then(r=>r.json())")
def until(page,predicate,seconds=4):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  s=state(page)
  if predicate(s): return s
  page.wait_for_timeout(50)
 raise AssertionError(state(page))

with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
 page=b.new_page(viewport={'width':1440,'height':1000})
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8080/',wait_until='domcontentloaded')
 page.wait_for_selector('.connection.online');page.wait_for_function("!document.querySelector('.loading')")
 page.keyboard.press('r');until(page,lambda s:s['time']<1)
 page.keyboard.down('w');until(page,lambda s:s['command'][0]>.09)
 page.wait_for_timeout(1500);walking=state(page);assert walking['position'][0]>.02
 page.keyboard.up('w');until(page,lambda s:s['command'][0]==0)
 page.keyboard.down('w');page.keyboard.down('a');until(page,lambda s:s['command'][0]>.09 and s['command'][2]>.5)
 page.keyboard.up('w');page.keyboard.up('a');until(page,lambda s:s['command']==[0,0,0])
 page.get_by_label('行走指令',exact=True).fill('0.03')
 page.get_by_label('行走指令',exact=True).press('Tab')
 page.keyboard.down('w');low=until(page,lambda s:abs(s['command'][0]-.03)<1e-6)
 assert low['active']=='walking'
 page.keyboard.up('w');until(page,lambda s:s['command'][0]==0)
 page.keyboard.down('w');until(page,lambda s:s['command'][0]>.02)
 page.evaluate("window.dispatchEvent(new Event('blur'))")
 until(page,lambda s:s['command'][0]==0)
 page.keyboard.up('w')
 recoveries=[]
 for key,pose in [('1','sitting'),('2','prone'),('3','supine'),('4','left_side'),('5','right_side')]:
  page.keyboard.press(key);until(page,lambda s:s['spawn']==pose and s['time']<1)
  recovered=until(page,lambda s:s['rise_time'] is not None and s['active']=='walking',seconds=9)
  recoveries.append({k:recovered[k] for k in ['spawn','rise_time','height','tilt']})
 page.keyboard.press('p');paused=until(page,lambda s:s['paused']);page.wait_for_timeout(400)
 assert state(page)['time']==paused['time']
 page.keyboard.press('p');until(page,lambda s:not s['paused'])
 page.get_by_label('控制模式',exact=True).select_option('recovery');until(page,lambda s:s['active']=='recovery')
 page.get_by_label('控制模式',exact=True).select_option('auto');until(page,lambda s:s['mode']=='auto')
 page.keyboard.press('r');until(page,lambda s:s['time']<1)
 page.wait_for_timeout(600)
 page.screenshot(path='simulator/logs/demo-desktop.png')
 page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(300)
 assert page.evaluate('document.documentElement.scrollWidth')<=390
 page.screenshot(path='simulator/logs/demo-mobile.png',full_page=True)
 assert not errors,errors
 result={'browser':'Chromium','keyboard':True,'release':True,'blur_stop':True,'low_speed_no_switch':True,'pause':True,'policy_switch':True,'walking_position_x':walking['position'][0],'recoveries':recoveries,'page_errors':errors}
 Path('simulator/logs/browser_check.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2));b.close()
