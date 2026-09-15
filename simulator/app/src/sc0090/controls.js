export function movement(keys, speed, turn) {
  const held = (...codes) => codes.some(code => keys.has(code)) ? 1 : 0;
  return [
    (held('KeyW', 'ArrowUp') - held('KeyS', 'ArrowDown')) * speed,
    (held('KeyQ') - held('KeyE')) * Math.min(speed, 0.15),
    (held('KeyA', 'ArrowLeft') - held('KeyD', 'ArrowRight')) * turn,
  ];
}

// Use the raw MuJoCo position and quaternion inside the upstream Z-up rig.
// In particular, do not clamp rendered joints: contact compliance can put
// actual qpos slightly beyond nominal joint limits.
export function syncPose(rig, state, rotation) {
  const trunk = rig.bodies.get('trunk_base');
  trunk.position.fromArray(state.position);
  const [w, x, y, z] = state.quaternion;
  trunk.quaternion.set(x, y, z, w);
  for (const [name, angle] of Object.entries(state.joints)) {
    const joint = rig.joints.get(name);
    if (!joint) throw new Error(`Visual model is missing joint ${name}`);
    rotation.setFromAxisAngle(joint.axis, angle);
    joint.body.quaternion.copy(joint.baseQuat).multiply(rotation);
  }
}
