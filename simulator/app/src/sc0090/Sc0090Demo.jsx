import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { buildRig, loadKinematics, MODEL_DIR } from '../game/duck.js';
import { movement, syncPose } from './controls.js';
import './style.css';

const NAMES = { auto: '自动切换', walking: '行走模型', recovery: '起身模型' };
const POSES = [['standing', '站立', 'R'], ['sitting', '坐姿', '1'], ['prone', '俯卧', '2'],
  ['supine', '仰卧', '3'], ['left_side', '左侧卧', '4'], ['right_side', '右侧卧', '5']];
const EVENT_KEYS = { KeyR: 'standing', Digit1: 'sitting', Digit2: 'prone', Digit3: 'supine',
  Digit4: 'left_side', Digit5: 'right_side', KeyF: 'push', KeyP: 'pause' };
const MOVE_KEYS = new Set(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ArrowUp','ArrowDown','ArrowLeft','ArrowRight']);

function Scene({ live, onReady, onError }) {
  const { scene, camera, gl } = useThree();
  const rig = useRef(null), orbit = useRef(null);
  const math = useRef({ rotation: new THREE.Quaternion(), target: new THREE.Vector3(), delta: new THREE.Vector3() });
  useEffect(() => {
    let disposed = false;
    scene.background = new THREE.Color('#ebe9e1');
    const pmrem = new THREE.PMREMGenerator(gl);
    const room = new RoomEnvironment();
    const env = pmrem.fromScene(room);
    scene.environment = env.texture;
    scene.environmentIntensity = 0.45;
    const control = new OrbitControls(camera, gl.domElement);
    control.enableDamping = true;
    control.target.set(0, .12, 0);
    control.minDistance = .35;
    control.maxDistance = 4;
    control.maxPolarAngle = Math.PI / 2 - .015;
    orbit.current = control;
    loadKinematics(`${MODEL_DIR}/kinematics.json`).then(buildRig).then(loaded => {
      if (disposed) return;
      rig.current = loaded;
      loaded.placer.traverse(object => { if (object.isMesh) object.castShadow = true; });
      scene.add(loaded.placer);
      onReady();
    }).catch(error => !disposed && onError(error.message));
    return () => {
      disposed = true;
      if (rig.current) {
        scene.remove(rig.current.placer);
        const materials = new Set();
        rig.current.placer.traverse(object => {
          if (object.material) materials.add(object.material);
        });
        materials.forEach(material => material.dispose());
      }
      rig.current = null;
      control.dispose(); env.dispose(); room.dispose(); pmrem.dispose();
    };
  }, [scene, camera, gl]);
  useFrame(() => {
    const state = live.current;
    if (rig.current && state?.position) {
      syncPose(rig.current, state, math.current.rotation);
      const [x,y,z] = state.position;
      math.current.target.set(x,z,-y);
      math.current.delta.copy(math.current.target).sub(orbit.current.target);
      camera.position.add(math.current.delta);
      orbit.current.target.copy(math.current.target);
    }
    orbit.current?.update();
  });
  return <>
    <ambientLight intensity={1.1}/>
    <directionalLight position={[2,4,2]} intensity={2.3} castShadow shadow-mapSize={[1024,1024]}
      shadow-camera-left={-2} shadow-camera-right={2} shadow-camera-top={2} shadow-camera-bottom={-2} shadow-bias={-.0001}/>
    <directionalLight position={[-2,1,-1]} intensity={.7}/>
    <mesh rotation={[-Math.PI/2,0,0]} position={[0,-.001,0]} receiveShadow>
      <planeGeometry args={[200,200]}/><meshStandardMaterial color="#ebe9e1" roughness={1}/>
    </mesh>
    <gridHelper args={[20,100,'#b4b0a3','#d1cdc1']} position={[0,.0001,0]}/>
  </>;
}

export default function Sc0090Demo() {
  const live = useRef(null), keys = useRef(new Set()), pending = useRef([]);
  const controls = useRef({ speed: .1, turn: .6, mode: 'auto' });
  const [state, setState] = useState(null), [ready, setReady] = useState(false);
  const [connected, setConnected] = useState(false), [error, setError] = useState('');
  const [busy, setBusy] = useState(false), [speed, setSpeed] = useState(.1);
  const [turn, setTurn] = useState(.6), [mode, setMode] = useState('auto');
  const [catalog, setCatalog] = useState({walking:[], recovery:[]});
  const [choices, setChoices] = useState({});
  const exploration = useRef(null);
  const [exploring, setExploring] = useState(false);
  const [exploreConfig, setExploreConfig] = useState({seed:1, duration:4, recovery_timeout:12, recoveries:true});
  const sendNow = useRef(() => {});
  function cancelExploration() {
    exploration.current = null;
    setExploring(false);
  }
  function toggleExploration() {
    keys.current.clear(); pending.current = [];
    if (exploration.current) cancelExploration();
    else {
      exploration.current = {...exploreConfig, id:crypto.randomUUID()};
      setExploring(true); setMode('auto'); controls.current.mode = 'auto';
    }
    sendNow.current();
  }
  function configureExploration(key, value) {
    cancelExploration(); keys.current.clear();
    setExploreConfig(previous => ({...previous,[key]:value})); sendNow.current();
  }
  async function refreshCatalog() {
    try {
      const response = await fetch('/api/checkpoints', {signal:AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error(`读取 checkpoint 列表失败：HTTP ${response.status}`);
      setCatalog(await response.json());
    } catch (e) { setError(e.message); }
  }
  useEffect(() => { refreshCatalog(); }, []);
  function loadCheckpoint(slot) {
    const id = choices[slot] || state?.selected_models?.[slot];
    if (!id) return;
    cancelExploration();
    keys.current.clear();
    if (pending.current.length < 16) pending.current.push({load:{slot,id}});
    sendNow.current();
  }
  function event(name) {
    cancelExploration();
    keys.current.clear();
    if (pending.current.length < 16) pending.current.push(name);
    sendNow.current();
  }
  useEffect(() => {
    let disposed = false, frameTimer, controlTimer, inFlight = false, seq = 0;
    const client = crypto.randomUUID();
    async function poll() {
      try {
        const response = await fetch('/api/frame', { signal: AbortSignal.timeout(1500), cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const next = await response.json();
        if (!next.position) throw new Error('等待仿真初始化');
        if (!disposed) {
          live.current = next; setState(next); setConnected(true);
          if (exploration.current?.id === next.exploration?.request_id && !next.exploration.active) cancelExploration();
        }
      } catch {
        if (!disposed) { setConnected(false); cancelExploration(); }
      }
      if (!disposed) frameTimer = setTimeout(poll, 30);
    }
    async function send() {
      if (inFlight || disposed) return;
      inFlight = true;
      const {speed, turn, mode} = controls.current;
      const action = pending.current.shift();
      try {
        const response = await fetch('/api/control', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          signal: AbortSignal.timeout(1000),
          body: JSON.stringify({client, seq: ++seq, mode,
            explore:exploration.current, explore_limits:[speed,turn],
            ...(typeof action === 'string' ? {event:action} : action || {}),
            twist: movement(keys.current, speed, turn)}),
        });
        if (!disposed) {
          setBusy(response.status === 409);
          if (!response.ok) cancelExploration();
          if (!response.ok && response.status !== 409) setError(`控制请求失败：HTTP ${response.status}`);
        }
      } catch {
        if (!disposed) { setConnected(false); cancelExploration(); }
      } finally { inFlight = false; }
    }
    function tick() {
      if (!document.hidden && document.hasFocus()) send();
      controlTimer = setTimeout(tick, 100);
    }
    const stop = () => { cancelExploration(); keys.current.clear(); pending.current = []; send(); };
    function keydown(e) {
      if (e.target.matches('input:not([type="range"]), select, textarea')) return;
      if (e.target.matches('input[type="range"]') && e.code.startsWith('Arrow')) return;
      if (MOVE_KEYS.has(e.code)) { e.preventDefault(); cancelExploration(); keys.current.add(e.code); send(); }
      else if (e.code === 'Space') { e.preventDefault(); stop(); }
      else if (EVENT_KEYS[e.code] && !e.repeat) { e.preventDefault(); event(EVENT_KEYS[e.code]); }
    }
    function keyup(e) { keys.current.delete(e.code); if (MOVE_KEYS.has(e.code)) send(); }
    sendNow.current = send;
    window.addEventListener('keydown', keydown);
    window.addEventListener('keyup', keyup);
    window.addEventListener('blur', stop);
    document.addEventListener('visibilitychange', stop);
    poll(); tick();
    return () => {
      stop(); disposed = true;
      clearTimeout(frameTimer); clearTimeout(controlTimer);
      window.removeEventListener('keydown', keydown);
      window.removeEventListener('keyup', keyup);
      window.removeEventListener('blur', stop);
      document.removeEventListener('visibilitychange', stop);
    };
  }, []);
  const value = (number, digits=2) => number == null ? '—' : number.toFixed(digits);
  return <main className="sc-demo">
    <header><div><span className="eyebrow">MICRODUCK / SC0090</span><h1>亲手试试，这只鸭子学会了什么。</h1></div>
      <span className={`connection ${connected ? 'online' : ''}`}>{connected ? '本地仿真已连接' : '正在连接本地仿真…'}</span></header>
    <section className="model-picker" aria-label="checkpoint 选择">
      {['walking','recovery'].map(slot => <div key={slot} className="model-choice">
        <label>{slot === 'walking' ? '行走' : '起身'} checkpoint
          <select aria-label={`${slot === 'walking' ? '行走' : '起身'} checkpoint`}
            value={choices[slot] || state?.selected_models?.[slot] || ''}
            onChange={e => {setChoices(previous => ({...previous,[slot]:e.target.value})); e.target.blur();}}>
            <option value="" disabled>选择本机训练的 checkpoint</option>
            {catalog[slot].map(entry => <option key={entry.id} value={entry.id}>
              #{entry.iteration} · {entry.family.replace('sc0090_','')} · {entry.run.slice(0,19).replace('_',' ')}{entry.ready ? ' · 已就绪' : ''}
            </option>)}
          </select>
        </label>
        <button aria-label={`加载${slot === 'walking' ? '行走' : '起身'} checkpoint`}
          disabled={!connected || state?.model_load?.state === 'loading' || !(choices[slot] || state?.selected_models?.[slot])}
          onClick={() => loadCheckpoint(slot)}>加载</button>
      </div>)}
      <div className="model-note"><span>{state?.model_load?.message || '选择并加载后重置为站立；编号属于各自训练运行。首次加载会在后台导出并缓存。'}</span>
        <button onClick={refreshCatalog}>刷新列表</button></div>
    </section>
    <section className="explore-panel" aria-label="自由探索">
      <div className="explore-title"><div><b>自由探索</b><p>随机组合行走、转弯、停步和扰动；相同序列编号可用于比较不同模型。</p></div>
        <button aria-label={exploring ? '停止自由探索' : '开始自由探索'}
          disabled={!connected || busy || state?.paused || !!state?.error || state?.model_load?.state === 'loading'}
          onClick={toggleExploration}>{exploring ? '停止探索' : '开始探索'}</button></div>
      <div className="explore-settings">
        <label>序列编号<input aria-label="探索序列编号" type="number" min="0" max="4294967295" step="1"
          value={exploreConfig.seed} onChange={e => configureExploration('seed', Math.max(0,Math.min(4294967295,Math.trunc(+e.target.value))))}/></label>
        <label>每段指令（秒）<input aria-label="探索动作时长" type="number" min="2" max="10" step="1"
          value={exploreConfig.duration} onChange={e => configureExploration('duration', Math.max(2,Math.min(10,+e.target.value)))}/></label>
        <label>起身等待上限（秒）<input aria-label="探索起身等待上限" type="number" min="2" max="60" step="1"
          value={exploreConfig.recovery_timeout} onChange={e => configureExploration('recovery_timeout', Math.max(2,Math.min(60,+e.target.value)))}/></label>
        <label className="explore-check"><input type="checkbox" checked={exploreConfig.recoveries}
          onChange={e => configureExploration('recoveries', e.target.checked)}/>包含倒地姿态重置与起身测试</label>
      </div>
      <p className={`explore-status ${state?.exploration?.failed ? 'failed' : ''}`}>
        {state?.exploration?.number > 0 ? `第 ${state.exploration.number} 段 · ` : ''}{state?.exploration?.label || '尚未开始'}
        {state?.exploration?.active ? ` · ${value(state.exploration.elapsed,1)} s` : ''}</p>
      <p className="hint">速度上限沿用控制台滑块。方向键、姿态按钮、窗口失焦或断连会退出探索；起身超时会暂停仿真并保留现场。</p>
      {state?.exploration?.history?.length > 0 && <ol className="explore-history">{state.exploration.history.map(item =>
        <li key={item.number}>#{item.number} {item.label} · {item.outcome}</li>)}</ol>}
    </section>
    <section className="stage" aria-label="机器人三维仿真">
      <Canvas shadows camera={{position:[.65,.4,.65],fov:42,near:.02,far:30}} dpr={[1,2]}>
        <Scene live={live} onReady={() => setReady(true)} onError={setError}/>
      </Canvas>
      <div className="stage-label"><span className="pill">{NAMES[state?.active] || '正在载入'}</span><span>SC0090 · 12 V · 80 rpm</span></div>
      {!ready && <div className="loading">正在加载三维机器人…</div>}
      <div className="stage-hint">拖动旋转视角 · 滚轮缩放 · 松开方向键停止指令</div>
      <div className="telemetry">
        <div><small>实际前进速度</small><strong>{value(state?.velocity?.[0])}<em>m/s</em></strong></div>
        <div><small>下发速度指令</small><strong>{value(state?.command?.[0])}<em>m/s</em></strong></div>
        <div><small>躯干倾斜</small><strong>{value(state?.tilt,1)}<em>°</em></strong></div>
        <div><small>起身并稳定用时</small><strong>{value(state?.rise_time)}<em>s</em></strong></div>
      </div>
    </section>
    <aside>
      <div className="panel-title"><span>控制台</span><small>策略 50 Hz · 实时物理</small></div>
      <label>控制模式<select aria-label="控制模式" value={mode} onChange={e => {
        cancelExploration();
        setMode(e.target.value); controls.current.mode=e.target.value; keys.current.clear(); sendNow.current(); e.target.blur();
      }}>{Object.entries(NAMES).map(([id,name]) => <option key={id} value={id}>{name}</option>)}</select></label>
      <p className="hint">自动模式在跌倒后调用起身模型，稳定站立后回到行走模型。</p>
      <label>行走指令 <output>{speed.toFixed(2)} m/s</output><input aria-label="行走指令" type="range" min="0.01" max="0.30" step="0.01" value={speed} onChange={e => {setSpeed(+e.target.value); controls.current.speed=+e.target.value;}}/></label>
      <label>转向指令 <output>{turn.toFixed(1)} rad/s</output><input aria-label="转向指令" type="range" min="0.1" max="1.0" step="0.1" value={turn} onChange={e => {setTurn(+e.target.value); controls.current.turn=+e.target.value;}}/></label>
      <div className="key-guide"><p><kbd>W</kbd> / <kbd>S</kbd> 前进、后退</p><p><kbd>A</kbd> / <kbd>D</kbd> 左转、右转</p><p><kbd>Q</kbd> / <kbd>E</kbd> 左移、右移</p><p><kbd>Space</kbd> 停止行走指令</p></div>
      <p className="group-label">从不同姿态开始</p>
      <div className="pose-buttons">{POSES.map(([id,name,key]) => <button key={id} onClick={() => event(id)}>{name}<kbd>{key}</kbd></button>)}</div>
      <div className="actions"><button onClick={() => event('push')}>侧向扰动 <kbd>F</kbd></button><button onClick={() => event('pause')}>{state?.paused ? '继续' : '暂停'} <kbd>P</kbd></button></div>
      <p className="hint">扰动为一次 0.2 m/s 侧向速度增量。起身时间包含连续稳定站立 0.5 秒；未达标时显示「—」。</p>
    </aside>
    <footer><div><b>当前模型</b><span>行走：{state?.models?.walking || '—'}　起身：{state?.models?.recovery || '—'}</span></div>
      <p>本地 CPU 仿真，使用训练时的 SC0090 BAM 电机模型。当前为固定参数演示，不等同于随机化评估或实机结果。</p>
      <a href="https://huggingface.co/spaces/pollen-robotics/microduck-simulator/tree/main" target="_blank" rel="noreferrer">官方网页源码 ↗</a></footer>
    {(error || state?.error || state?.model_load?.state === 'error' || busy || !connected) && <div role="status" className="notice">{error || state?.error || (state?.model_load?.state === 'error' ? state.model_load.message : null) || (busy ? '另一个窗口正在控制。关闭该窗口或切到后台后，本页即可接管。' : '仿真连接中断：行走指令会自动归零。请检查本地服务。')}</div>}
  </main>;
}
