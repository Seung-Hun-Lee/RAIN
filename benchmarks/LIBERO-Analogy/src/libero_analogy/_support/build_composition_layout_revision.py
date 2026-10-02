import numpy as np
def bounds(env, name):
    """Conservative collision-geometry world bounds, not visual-mesh bounds."""
    lo, hi = [], []
    obj = env.env.get_object(name)
    for geom in obj.contact_geoms:
        gid = env.sim.model.geom_name2id(geom)
        center = env.sim.data.geom_xpos[gid]
        rot = env.sim.data.geom_xmat[gid].reshape(3, 3)
        size = env.sim.model.geom_size[gid]
        kind = int(env.sim.model.geom_type[gid])
        if kind == 2:  # sphere
            extent = np.full(3, size[0])
        elif kind == 3:  # capsule
            extent = np.abs(rot[:, 2]) * size[1] + size[0]
        elif kind == 5:  # cylinder
            extent = np.sqrt(np.maximum(0, 1 - rot[:, 2] ** 2)) * size[0] + np.abs(rot[:, 2]) * size[1]
        else:
            extent = np.abs(rot) @ size
        lo.append(center - extent)
        hi.append(center + extent)
    if not lo:
        bid = env.sim.model.body_name2id(obj.root_body)
        pos = env.sim.data.body_xpos[bid].copy()
        return pos, pos
    return np.min(lo, axis=0), np.max(hi, axis=0)
