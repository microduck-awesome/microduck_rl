// Diagnostic port of mujoco-warp 3.8.1 plane_convex.
// Copyright 2025 The Newton Developers, Apache-2.0.
// Registration is model-scoped; unrelated models/geometries use native MuJoCo.
// The 1e-3 support window below selects candidate vertices; it is not a pad
// thickness. Only candidates with dist < margin become contacts.
// Install the dispatcher before physics workers start. Per-call geometry is
// stack-local; the registry mutex is released before either collision routine.
#include <algorithm>
#include <array>
#include <cmath>
#include <mujoco/mujoco.h>
#include <mutex>
#include <unordered_map>
struct V {
  float x, y, z;
};
static V add(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V scale(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V cross(V a, V b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static V load(const mjtNum *p) {
  return {(float)p[0], (float)p[1], (float)p[2]};
}
static V mul(const float *r, V a) {
  return {dot({r[0], r[1], r[2]}, a), dot({r[3], r[4], r[5]}, a),
          dot({r[6], r[7], r[8]}, a)};
}
static V tmul(const float *r, V a) {
  return {dot({r[0], r[3], r[6]}, a), dot({r[1], r[4], r[7]}, a),
          dot({r[2], r[5], r[8]}, a)};
}
static int collision(const mjModel *m, mjData *d, mjPreContact *con, int g1,
                     int g2, mjtNum margin) {
  const int mesh = m->geom_dataid[g2], adr = m->mesh_vertadr[mesh],
            num = m->mesh_vertnum[mesh];
  const auto vertex = [&](int i) -> V {
    const float *v = m->mesh_vert + 3 * (adr + i);
    return {v[0], v[1], v[2]};
  };
  float r[9];
  for (int i = 0; i < 9; ++i)
    r[i] = (float)d->geom_xmat[9 * g2 + i];
  V normal = {(float)d->geom_xmat[9 * g1 + 2], (float)d->geom_xmat[9 * g1 + 5],
              (float)d->geom_xmat[9 * g1 + 8]};
  V pos = load(d->geom_xpos + 3 * g2), plane = load(d->geom_xpos + 3 * g1);
  V local = tmul(r, sub(plane, pos)), n = tmul(r, normal);
  const auto support = [&](int i) { return dot(sub(local, vertex(i)), n); };
  constexpr float huge = 1e6f;
  int indices[4] = {-1, -1, -1, -1};
  V a{}, b{}, c{};
  const int graphadr = m->mesh_graphadr[mesh];
  if (graphadr < 0 || num < 10) {
    float maximum = -huge;
    for (int i = 0; i < num; ++i) {
      float v = support(i);
      if (v > maximum) {
        maximum = v;
        indices[0] = i;
        a = vertex(i);
      }
    }
    if (maximum < 0)
      return 0;
    float threshold = maximum - 1e-3f;
    auto mask = [&](int i) { return support(i) > threshold ? 0.f : -huge; };
    float best = -huge;
    for (int i = 0; i < num; ++i) {
      V delta = sub(a, vertex(i));
      float value = dot(delta, delta) + mask(i);
      if (value > best) {
        best = value;
        indices[1] = i;
        b = vertex(i);
      }
    }
    V ab = cross(n, sub(a, b));
    best = -huge;
    for (int i = 0; i < num; ++i) {
      float value = std::abs(dot(sub(a, vertex(i)), ab)) + mask(i);
      if (value > best) {
        best = value;
        indices[2] = i;
        c = vertex(i);
      }
    }
    V ac = cross(n, sub(a, c)), bc = cross(n, sub(b, c));
    best = -huge;
    for (int i = 0; i < num; ++i) {
      float value = (std::abs(dot(sub(a, vertex(i)), ac)) + mask(i)) +
                    (std::abs(dot(sub(b, vertex(i)), bc)) + mask(i));
      if (value > best) {
        best = value;
        indices[3] = i;
      }
    }
  } else {
    const int *graph = m->mesh_graph + graphadr;
    int nv = graph[0];
    const int *edgeadr = graph + 2;
    const int *global = edgeadr + nv;
    const int *edges = global + nv;
    int imax = 0;
    auto climb = [&](auto value) {
      float best = -huge;
      int previous;
      do {
        previous = imax;
        int j = edgeadr[imax];
        while (edges[j] >= 0) {
          int neighbor = edges[j++];
          float score = value(global[neighbor]);
          if (score > best) {
            best = score;
            imax = neighbor;
          }
        }
      } while (previous != imax);
      return best;
    };
    float maximum = climb(support), threshold = std::max(0.f, maximum - 1e-3f);
    auto mask = [&](int i) { return support(i) > threshold ? 0.f : -huge; };
    climb([&](int i) {
      float v = support(i);
      return v > threshold ? v : -huge;
    });
    indices[0] = global[imax];
    a = vertex(indices[0]);
    climb([&](int i) {
      V delta = sub(a, vertex(i));
      return dot(delta, delta) + mask(i);
    });
    indices[1] = global[imax];
    b = vertex(indices[1]);
    V ab = cross(n, sub(a, b));
    climb(
        [&](int i) { return std::abs(dot(sub(a, vertex(i)), ab)) + mask(i); });
    indices[2] = global[imax];
    c = vertex(indices[2]);
    V ac = cross(n, sub(a, c)), bc = cross(n, sub(b, c));
    climb([&](int i) {
      return (std::abs(dot(sub(a, vertex(i)), ac)) + mask(i)) +
             (std::abs(dot(sub(b, vertex(i)), bc)) + mask(i));
    });
    indices[3] = global[imax];
  }
  int count = 0;
  for (int i = 3; i >= 0; --i) {
    int idx = indices[i], duplicates = 0;
    for (int j = 0; j <= i; ++j)
      duplicates += (indices[j] == idx);
    if (duplicates != 1 || idx < 0)
      continue;
    float dist = -support(idx);
    if (!(dist < (float)margin))
      continue;
    V point = sub(add(pos, mul(r, vertex(idx))), scale(normal, 0.5f * dist));
    mjPreContact &out = con[count++];
    out.dist = dist;
    out.pos[0] = point.x;
    out.pos[1] = point.y;
    out.pos[2] = point.z;
    out.normal[0] = normal.x;
    out.normal[1] = normal.y;
    out.normal[2] = normal.z;
    out.tangent[0] = out.tangent[1] = out.tangent[2] = 0;
  }
  return count;
}
static std::mutex lock;
static std::unordered_map<const mjModel *, std::array<int, 2>> models;
static mjfCollision original = nullptr;
static int dispatch(const mjModel *m, mjData *d, mjPreContact *con, int g1,
                    int g2, mjtNum margin) {
  bool use = false;
  {
    std::lock_guard<std::mutex> guard(lock);
    auto it = models.find(m);
    if (it != models.end())
      use = (g2 == it->second[0] || g2 == it->second[1]);
  }
  return use ? collision(m, d, con, g1, g2, margin)
             : original(m, d, con, g1, g2, margin);
}
extern "C" int register_model(const mjModel *m, int left, int right) {
  std::lock_guard<std::mutex> guard(lock);
  if (!original) {
    original = mjCOLLISIONFUNC[mjGEOM_PLANE][mjGEOM_MESH];
    mjCOLLISIONFUNC[mjGEOM_PLANE][mjGEOM_MESH] = dispatch;
  }
  if (mjCOLLISIONFUNC[mjGEOM_PLANE][mjGEOM_MESH] != dispatch || models.count(m))
    return 1;
  models[m] = {left, right};
  return 0;
}
extern "C" void unregister_model(const mjModel *m) {
  std::lock_guard<std::mutex> guard(lock);
  models.erase(m);
}
