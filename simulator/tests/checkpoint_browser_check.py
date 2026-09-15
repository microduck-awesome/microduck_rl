"""Exercise actual UI selection, uncached CPU export, cached reload and rollback."""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright


def state(page):
    return page.evaluate("fetch('/api/frame').then(r=>r.json())")


def until(page, predicate, timeout=150):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        result = state(page)
        if predicate(result):
            return result
        if result.get('model_load',{}).get('state') == 'error':
            raise AssertionError(result['model_load'])
        page.wait_for_timeout(150)
    raise AssertionError(state(page))


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    page = browser.new_page(viewport={'width':1440,'height':1100})
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:8080/',wait_until='domcontentloaded')
    page.wait_for_selector('.connection.online')
    page.wait_for_function("!document.querySelector('.loading')")
    catalog=page.evaluate("fetch('/api/checkpoints').then(r=>r.json())")
    original=state(page)['selected_models']
    report=[]
    for slot,family,iteration in [('walking','sc0090_walk',4050),('recovery','sc0090_recovery_v3',5300)]:
        entry=next(e for e in catalog[slot] if e['family']==family and e['iteration']==iteration)
        label='行走' if slot=='walking' else '起身'
        before=state(page)['time']; start=time.monotonic()
        page.get_by_label(label+' checkpoint',exact=True).select_option(entry['id'])
        page.get_by_role('button',name='加载'+label+' checkpoint',exact=True).click()
        result=until(page,lambda s:s['selected_models'][slot]==entry['id'])
        assert result['spawn']=='standing' and result['time']<2
        assert f'model_{iteration}_' in result['models'][slot]
        report.append({'slot':slot,'checkpoint':entry['checkpoint'],'seconds':time.monotonic()-start,'selected':entry['id']})
        page.get_by_label(label+' checkpoint',exact=True).select_option(original[slot])
        page.get_by_role('button',name='加载'+label+' checkpoint',exact=True).click()
        until(page,lambda s:s['selected_models'][slot]==original[slot],10)
        # The same earlier checkpoint now loads from the verified ONNX cache.
        start=time.monotonic()
        page.get_by_label(label+' checkpoint',exact=True).select_option(entry['id'])
        page.get_by_role('button',name='加载'+label+' checkpoint',exact=True).click()
        until(page,lambda s:s['selected_models'][slot]==entry['id'],10)
        report[-1]['cached_seconds']=time.monotonic()-start
        page.get_by_label(label+' checkpoint',exact=True).select_option(original[slot])
        page.get_by_role('button',name='加载'+label+' checkpoint',exact=True).click()
        until(page,lambda s:s['selected_models'][slot]==original[slot],10)
    page.get_by_role('button',name='刷新列表').click()
    page.screenshot(path='simulator/logs/checkpoint-picker.png',full_page=True)
    assert not errors,errors
    Path('simulator/logs/checkpoint_browser_check.json').write_text(json.dumps({'passed':True,'cases':report,'page_errors':errors},indent=2))
    print(json.dumps(report,indent=2))
    browser.close()
