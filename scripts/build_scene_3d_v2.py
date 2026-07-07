"""Blender warehouse scene v2 - CC0 assets upgrade (M3-v2).

Assets (all recorded in the license log):
  Forklift.glb / Worker.glb / Wooden Pallet.glb  - poly.pizza (CC0)
  corrugated_iron_02 (wall PBR), gravel_embedded_concrete (floor PBR),
  industrial_workshop_foundry_2k.hdr (HDRI)      - polyhaven.com (CC0)

Camera, cam_target and the worker_root motion path are IDENTICAL to v1, so
src/config.py zone geometry and the whole A layer stay untouched.

Run:  blender -b -noaudio --python scripts/build_scene_3d_v2.py -- --still 264 out.png
      blender -b -noaudio --python scripts/build_scene_3d_v2.py -- --anim out_dir 1 384 3 [budget_s]
      (open the saved .blend in Blender GUI for hi-quality renders)
"""
import math
import os
import sys

import bpy

FPS, DUR = 24, 16.0
N_FRAMES = int(FPS * DUR)
RES_X, RES_Y = 480, 270
SAMPLES = 8
ASSETS = os.environ.get("VLM_ASSETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "downloads"))

# ---------- helpers ----------

def mat(name, color, rough=0.6, metal=0.0, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emit
    return m


def pbr_mat(name, diff, rough=None, nor=None, scale=1.0):
    """Box-projected PBR material (no UV needed)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def img_node(path, non_color=False):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path)
        n.projection = "BOX"
        n.projection_blend = 0.2
        if non_color:
            try:
                n.image.colorspace_settings.name = "Non-Color"
            except TypeError:  # sandbox blender lacks full OCIO config
                n.image.colorspace_settings.name = "Linear"
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    d = img_node(diff)
    nt.links.new(d.outputs["Color"], bsdf.inputs["Base Color"])
    if rough:
        r = img_node(rough, non_color=True)
        nt.links.new(r.outputs["Color"], bsdf.inputs["Roughness"])
    if nor:
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = 0.6
        n = img_node(nor, non_color=True)
        nt.links.new(n.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def box(name, size, loc, material, rot=(0, 0, 0)):
    # base cube edge=2 (half-extent 1) so scale=size/2 yields TRUE dimensions
    bpy.ops.mesh.primitive_cube_add(size=2, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    if material:
        o.data.materials.append(material)
    return o


def world_bbox(objs):
    from mathutils import Vector
    bpy.context.view_layer.update()
    pts = []
    for o in objs:
        if o.type == "MESH":
            pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def normalize(root_obj, objs, target, axis=2, drop_to_floor=True):
    """Scale hierarchy so its size along `axis` == target; put its base on z=0."""
    lo, hi = world_bbox(objs)
    size = (hi - lo)[axis]
    if size > 1e-6:
        f = target / size
        root_obj.scale = tuple(sc * f for sc in root_obj.scale)
    if drop_to_floor:
        lo, hi = world_bbox(objs)
        root_obj.location.z -= lo.z
    bpy.context.view_layer.update()


def place_at(root_obj, objs, x, y):
    """Move hierarchy so its bbox center (x,y) lands at target (handles glbs with offset origins)."""
    lo, hi = world_bbox(objs)
    cx, cy = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2
    root_obj.location.x += x - cx
    root_obj.location.y += y - cy
    bpy.context.view_layer.update()


def import_glb(path, wrapper_name):
    """Import a glb and parent ALL its root objects under one wrapper empty."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.data.objects if o not in before]
    bpy.ops.object.empty_add(location=(0, 0, 0))
    wrap = bpy.context.object
    wrap.name = wrapper_name
    for o in objs:
        if o.parent is None:
            o.parent = wrap
    return wrap, objs


# ---------- reset ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.engine = "CYCLES"
scn.cycles.device = "CPU"
scn.cycles.samples = SAMPLES
scn.cycles.use_denoising = False   # sandbox OIDN broken; enable on desktop GUI
scn.cycles.max_bounces = 4
scn.render.resolution_x, scn.render.resolution_y = RES_X, RES_Y
scn.render.fps = FPS
scn.frame_start, scn.frame_end = 1, N_FRAMES

# ---------- materials ----------
m_floor = pbr_mat(
    "floor_pbr",
    os.path.join(ASSETS, "gravel_embedded_concrete_diff_2k.jpg"),
    os.path.join(ASSETS, "gravel_embedded_concrete_rough_2k.jpg"),
    os.path.join(ASSETS, "gravel_embedded_concrete_nor_gl_2k.jpg"),
    scale=0.35,
)
WALLT = os.path.join(ASSETS, "corrugated_iron_02_2k.blend", "textures")
m_wall = pbr_mat(
    "wall_pbr",
    os.path.join(WALLT, "corrugated_iron_02_diff_2k.jpg"),
    os.path.join(WALLT, "corrugated_iron_02_rough_2k.exr"),
    os.path.join(WALLT, "corrugated_iron_02_nor_gl_2k.exr"),
    scale=0.55,
)
m_blue = mat("stripe_blue", (0.04, 0.13, 0.40), rough=0.6)
m_dark = mat("opening", (0.045, 0.045, 0.05), rough=0.9)
m_slat = mat("door_slat", (0.50, 0.51, 0.50), rough=0.4, metal=0.6)
m_frame = mat("door_frame", (0.38, 0.38, 0.38), rough=0.5)
m_yellow = mat("hazard_y", (0.80, 0.60, 0.02), rough=0.6)
m_black = mat("hazard_b", (0.03, 0.03, 0.03), rough=0.6)
m_card = mat("cardboard", (0.40, 0.25, 0.12), rough=0.8)
m_card2 = mat("cardboard2", (0.46, 0.31, 0.16), rough=0.8)

# ---------- floor & wall ----------
box("floor", (30, 20, 0.1), (0, 5, -0.05), m_floor)

# ---------- wall with REAL openings (hollowed doorways) ----------
DX = 0.9
dw, dh = 2.9, 3.1            # doorway A (forklift): x -0.55..2.35, z 0..3.1
BX = -3.3
bw_, bh_ = 2.3, 2.5          # doorway B (boxes):   x -4.45..-2.15, z 0..2.5
WY, WT, WH = -0.15, 0.3, 6.2  # wall y-center, thickness, height (z -0.2..6)

box("wall_L", (15 - 4.45, WT, WH), ((-15 - 4.45) / 2, WY, 2.9), m_wall)
box("wall_mid", (4.45 - 2.15 - 0.55 + 0.55 - 1.6 + 1.6, WT, WH), (0, 0, 0), m_wall)  # placeholder, fixed below
bpy.data.objects.remove(bpy.data.objects["wall_mid"], do_unlink=True)
box("wall_M", (1.6, WT, WH), (-1.35, WY, 2.9), m_wall)                 # between the two doors
box("wall_R", (15 - 2.35, WT, WH), ((15 + 2.35) / 2, WY, 2.9), m_wall)
box("wall_headA", (dw, WT, 6.0 - dh), (DX, WY, (6.0 + dh) / 2), m_wall)
box("wall_headB", (bw_, WT, 6.0 - bh_), (BX, WY, (6.0 + bh_) / 2), m_wall)

# blue stripe (skips doorway A opening; z2.85 is above door B top so it continues there)
box("stripe_L", (14.45, 0.02, 0.45), ((-15 - 0.55) / 2, 0.012, 2.85), m_blue)
box("stripe_R", (12.65, 0.02, 0.45), ((15 + 2.35) / 2, 0.012, 2.85), m_blue)

def interior_room(name, cx, w, h, depth=2.2):
    """Dark room shell behind an opening - full wall height so key light can't flood in."""
    SH = 6.4  # shell height
    box(f"{name}_back", (w + 0.6, 0.1, SH), (cx, -depth, SH / 2 - 0.2), m_dark)
    box(f"{name}_sideL", (0.1, depth, SH), (cx - w / 2 - 0.05, -depth / 2, SH / 2 - 0.2), m_dark)
    box(f"{name}_sideR", (0.1, depth, SH), (cx + w / 2 + 0.05, -depth / 2, SH / 2 - 0.2), m_dark)
    box(f"{name}_top", (w + 0.6, depth, 0.1), (cx, -depth / 2, 6.0), m_dark)

interior_room("roomA", DX, dw, dh, depth=2.7)
interior_room("roomB", BX, bw_, bh_, depth=1.6)

# door trim
box("lintelA", (dw + 0.5, 0.06, 0.32), (DX, 0.032, dh + 0.16), m_frame)
for i in range(6):
    box(f"slatA{i}", (dw + 0.3, 0.07, 0.085), (DX, 0.05, dh + 0.36 + i * 0.095), m_slat)
box("jambL", (0.14, 0.12, dh), (DX - dw / 2 - 0.08, 0.035, dh / 2), m_frame)
box("jambR", (0.14, 0.12, dh), (DX + dw / 2 + 0.08, 0.035, dh / 2), m_frame)
box("lintelB", (bw_ + 0.5, 0.06, 0.3), (BX, 0.032, bh_ + 0.15), m_frame)
box("jambBL", (0.14, 0.12, bh_), (BX - bw_ / 2 - 0.08, 0.035, bh_ / 2), m_frame)
box("jambBR", (0.14, 0.12, bh_), (BX + bw_ / 2 + 0.08, 0.035, bh_ / 2), m_frame)
for i, (w, d, h) in enumerate([(0.8, 0.55, 0.5), (0.7, 0.5, 0.45), (0.6, 0.45, 0.4)]):
    box(f"boxin{i}", (w, d, h), (BX + 0.1, 0.12, 0.25 + i * 0.47), m_card if i % 2 else m_card2)

# ---------- hazard line ----------
for i in range(-15, 16):
    box(f"hz{i}", (0.5, 0.15, 0.104), (i * 0.5, 4.3, -0.048), m_yellow if i % 2 else m_black)

# ---------- forklift (poly.pizza CC0) ----------
fk_root, fk = import_glb(os.path.join(ASSETS, "Forklift.glb"), "forklift_wrap")
fk_root.rotation_euler = (0, 0, math.radians(180))  # face +Y (camera)
normalize(fk_root, fk, 2.2, axis=2)
place_at(fk_root, fk, DX, -0.35)

# --- forklift motion synced to the demo story ---
# efficient: slowly drives out | person in zone: HALTS | after clear: resumes
def fkey(fr, dy):
    fk_root.location.y = fk_base_y + dy
    fk_root.keyframe_insert("location", index=1, frame=fr)

fk_base_y = fk_root.location.y
fkey(1, 0.0)
fkey(int(9.5 * FPS), 0.40)      # creeping forward while clear/approaching
fkey(int(10.2 * FPS), 0.45)     # decelerates as person nears the zone
fkey(int(13.0 * FPS), 0.45)     # HALTED by the alert while person is inside the zone
fkey(N_FRAMES, 0.75)            # resumes after zone is clear
for fc in fk_root.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

# ---------- pallets (poly.pizza CC0) + cardboard boxes ----------
pl_root, pl = import_glb(os.path.join(ASSETS, "Wooden Pallet.glb"), "pallet_wrap")
pl_root.rotation_euler = (0, 0, math.radians(15))
normalize(pl_root, pl, 1.25, axis=0)
place_at(pl_root, pl, -3.45, 3.15)
import random
random.seed(7)
for i in range(2):
    for j in range(2):
        for k in range(2):
            w = 0.5 + random.uniform(-0.04, 0.04)
            box(f"pbox{i}{j}{k}", (w, 0.46, 0.4),
                (-3.45 - 0.28 + i * 0.58, 3.15 - 0.24 + j * 0.5, 0.39 + k * 0.41),
                m_card if (i + j + k) % 2 else m_card2)

# ---------- worker (poly.pizza CC0, rigged + Walk/Idle actions) ----------
bpy.ops.object.empty_add(location=(-8.5, 2.7, 0))
root = bpy.context.object
root.name = "worker_root"

wk_root, wk = import_glb(os.path.join(ASSETS, "Worker.glb"), "worker_wrap")
arm = next(o for o in wk if o.type == "ARMATURE")
normalize(wk_root, [o for o in wk if o.type == "MESH"], 1.8, axis=2)
wk_root.parent = root
wk_root.location = (0, 0, 0)
wk_root.rotation_euler = (0, 0, math.radians(90))  # compensate model forward axis

def key(fr, x, y):
    root.location = (x, y, 0)
    root.keyframe_insert("location", frame=fr)

# path curves around the pallet stack (pallet ~x -4.0..-2.9, y 2.6..3.75)
key(1, -8.5, 2.9)
key(int(5.0 * FPS), -8.5, 2.9)
key(int(6.3 * FPS), -6.2, 2.6)
key(int(7.3 * FPS), -4.6, 1.95)   # pass the aisle behind the pallet (upper body visible over boxes)
key(int(8.3 * FPS), -2.6, 1.95)
key(int(9.5 * FPS), -1.4, 2.5)    # rejoin toward the operating zone
key(int(10.7 * FPS), 0.6, 3.05)    # stops at the zone edge, ~1.2m from the fork tips
key(int(12.0 * FPS), 0.65, 3.05)   # (hears the alarm / sees the halted forklift, then leaves)
key(int(13.2 * FPS), -1.6, 2.4)
key(int(14.2 * FPS), -3.0, 1.95)  # same aisle on the way back
key(int(15.0 * FPS), -4.8, 1.95)
key(N_FRAMES, -8.5, 2.7)
for fc in root.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

def yaw(fr, deg):
    root.rotation_euler = (0, 0, math.radians(deg))
    root.keyframe_insert("rotation_euler", frame=fr)

yaw(1, 0)                      # walking toward +X
yaw(int(12.0 * FPS), 0)
yaw(int(12.5 * FPS), 180)      # walking back toward -X

# NLA: Walk while moving, Idle while standing in the zone
walk = bpy.data.actions.get("CharacterArmature|Walk_CharacterArmature")
idle = bpy.data.actions.get("CharacterArmature|Idle_Neutral_CharacterArmature")
if arm.animation_data is None:
    arm.animation_data_create()
arm.animation_data.action = None
track = arm.animation_data.nla_tracks.new()

def strip(action, start, end):
    s = track.strips.new(action.name, int(start), action)
    length = action.frame_range[1] - action.frame_range[0]
    s.repeat = max(1.0, (end - start) / length)
    s.frame_end = int(end)
    return s

strip(walk, 1, int(10.7 * FPS))
strip(idle, int(10.7 * FPS) + 1, int(12.0 * FPS))
strip(walk, int(12.0 * FPS) + 1, N_FRAMES)

# ---------- lights & HDRI world ----------
scn.world = bpy.data.worlds.new("world")
scn.world.use_nodes = True
wnt = scn.world.node_tree
env = wnt.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(os.path.join(ASSETS, "industrial_workshop_foundry_2k.hdr"))
bg = wnt.nodes["Background"]
bg.inputs["Strength"].default_value = 0.65
wnt.links.new(env.outputs["Color"], bg.inputs["Color"])

def arealight(name, loc, power, size=4, rot=None):
    bpy.ops.object.light_add(type="AREA", location=loc)
    L = bpy.context.object
    L.name = name
    L.data.energy = power
    L.data.size = size
    if rot:
        L.rotation_euler = rot
    return L

arealight("key", (0, 5.5, 6.5), 1000, 7)
arealight("wall_wash", (0, 7.0, 4.8), 1700, 8, rot=(math.radians(-75), 0, 0))

# ---------- camera (identical to v1 - do not change) ----------
bpy.ops.object.camera_add(location=(-7.5, 7.2, 4.3))
cam = bpy.context.object
cam.data.lens = 33
bpy.ops.object.empty_add(location=(0.5, 1.6, 0.6))
target = bpy.context.object
target.name = "cam_target"
tc = cam.constraints.new("TRACK_TO")
tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"
scn.camera = cam

import bpy_extras
from mathutils import Vector
scn.frame_set(1)
import numpy as np
def proj(p):
    v = bpy_extras.object_utils.world_to_camera_view(scn, cam, Vector(p))
    return np.array([v.x, 1 - v.y])
C = proj((0.9, 2.4, 0))
u = proj((0.9 + 1.6, 2.4, 0)) - C   # conjugate semi-diameters of the projected ellipse
v = proj((0.9, 2.4 + 1.15, 0)) - C
M = np.column_stack([u, v])
U, S, _ = np.linalg.svd(M)
ang = float(np.degrees(np.arctan2(U[1, 0], U[0, 0])))
print(f"ZONE_SCREEN center=({C[0]:.3f},{C[1]:.3f}) axes=({S[0]:.3f},{S[1]:.3f}) angle={ang:.1f}")

# ---------- CLI ----------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if argv and argv[0] == "--still":
    scn.frame_set(int(argv[1]))
    scn.render.filepath = argv[2]
    bpy.ops.render.render(write_still=True)
elif argv and argv[0] == "--anim":
    import time
    out_dir, start, end, step = argv[1], int(argv[2]), int(argv[3]), int(argv[4])
    budget = float(argv[5]) if len(argv) > 5 else 1e9
    t0 = time.time()
    for fr in range(start, end + 1, step):
        fp = f"{out_dir}/frame_{fr:04d}.png"
        if os.path.exists(fp):
            continue
        if time.time() - t0 > budget:
            print(f"BUDGET-STOP before frame {fr}")
            break
        scn.frame_set(fr)
        scn.render.filepath = fp
        bpy.ops.render.render(write_still=True)
    print("ANIM-DONE-THROUGH", fr)
elif argv and argv[0] == "--save":
    # desktop-ready: hi-quality settings + pack all textures into the .blend
    scn.cycles.samples = 128
    scn.cycles.use_denoising = True
    scn.render.resolution_x, scn.render.resolution_y = 960, 540
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=argv[1])
