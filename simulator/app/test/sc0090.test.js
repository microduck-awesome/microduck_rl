import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { movement, syncPose } from '../src/sc0090/controls.js';

test('low speed commands, simultaneous keys, releases and turn signs', () => {
  assert.deepEqual(movement(new Set(['KeyW','KeyA']), .03, .6), [.03,0,.6]);
  assert.deepEqual(movement(new Set(['KeyW','KeyS']), .1, .6), [0,0,0]);
  assert.deepEqual(movement(new Set(['KeyE','KeyD']), .3, .6), [0,-.15,-.6]);
  assert.deepEqual(movement(new Set(), .1, .6), [0,0,0]);
});

test('render raw physical pose without limit clipping or quaternion reordering bugs', () => {
  const trunk = new THREE.Group(), body = new THREE.Group();
  const joint = {body, axis:new THREE.Vector3(0,1,0), baseQuat:new THREE.Quaternion(), range:[-.5,.5]};
  const rig = {bodies:new Map([['trunk_base',trunk]]), joints:new Map([['knee',joint]])};
  syncPose(rig, {position:[1,2,3],quaternion:[.5,.5,.5,.5],joints:{knee:.6}}, new THREE.Quaternion());
  assert.deepEqual(trunk.position.toArray(),[1,2,3]);
  assert.deepEqual(trunk.quaternion.toArray(),[.5,.5,.5,.5]);
  assert.ok(body.quaternion.angleTo(new THREE.Quaternion().setFromAxisAngle(joint.axis,.6)) < 1e-6);
});
