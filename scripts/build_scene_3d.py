"""Blender (3.0+) procedural warehouse scene - VLM Scene Advisor M3 (B layer).

Generic autonomous pallet-forklift in a doorway (screen center-left), boxes/pallet
(screen right), hazard line, worker in orange vest entering the operating zone.
Timeline (16s @ 24fps) mirrors data/sample_clip.mp4:
  0-5s empty | 5-9.5s approach | 9.5-12s in zone | 12-16s leave

Camera looks from +Y toward the wall; screen-right = world -X.

Run:  blender -b -noaudio --python scripts/build_scene_3d.py -- --still 264 out.png
      blender -b -noaudio --python scripts/build_scene_3d.py -- --anim out_dir start end step
NOTE: OIDN denoising renders black in the dev sandbox -> denoise off; use
scripts/render_3d_clip.py (OpenCV post-denoise) instead.
"""
import math
import sys

import bpy

FPS, DUR = 24, 16.0
N_FRAMES = int(FPS * DUR)
RES_X, RES_Y = 480, 270
SAMPLES = 8

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


def box(name, size, loc, material, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    o.data.materials.append(material)
    return o


def cyl(name, r, depth, loc, material, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o


def sphere(name, r, loc, material, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=24, ring_count=16)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return o


# ---------- reset ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.engine = "CYCLES"
scn.cycles.device = "CPU"
scn.cycles.samples = SAMPLES
scn.cycles.use_denoising = False  # OIDN broken in sandbox; denoise in post
scn.cycles.max_bounces = 4
scn.render.resolution_x, scn.render.resolution_y = RES_X, RES_Y
scn.render.fps = FPS
scn.frame_start, scn.frame_end = 1, N_FRAMES

# ---------- materials ----------
m_floor = mat("floor", (0.28, 0.28, 0.29), rough=0.35)
m_floor2 = mat("floor_apron", (0.20, 0.20, 0.21), rough=0.45)
m_wall = mat("wall", (0.68, 0.68, 0.66), rough=0.7)
m_seam = mat("seam", (0.56, 0.56, 0.54), rough=0.7)
m_blue = mat("stripe_blue", (0.04, 0.13, 0.40), rough=0.6)
m_dark = mat("opening", (0.045, 0.045, 0.05), rough=0.9)
m_slat = mat("door_slat", (0.50, 0.51, 0.50), rough=0.4, metal=0.6)
m_frame = mat("door_frame", (0.38, 0.38, 0.38), rough=0.5)
m_yellow = mat("hazard_y", (0.80, 0.60, 0.02), rough=0.6)
m_black = mat("hazard_b", (0.03, 0.03, 0.03), rough=0.6)
m_red = mat("forklift_red", (0.42, 0.02, 0.03), rough=0.35, metal=0.15)
m_darkm = mat("forklift_dark", (0.09, 0.09, 0.10), rough=0.5, metal=0.4)
m_grey = mat("forklift_grey", (0.50, 0.50, 0.51), rough=0.35, metal=0.7)
m_screen = mat("screen", (0.05, 0.25, 0.55), rough=0.2, emit=2.0)
m_sensor = mat("sensor", (0.02, 0.02, 0.02), rough=0.2)
m_card = mat("cardboard", (0.40, 0.25, 0.12), rough=0.8)
m_card2 = mat("cardboard2", (0.46, 0.31, 0.16), rough=0.8)
m_pallet = mat("pallet", (0.33, 0.20, 0.09), rough=0.8)
m_vest = mat("vest", (1.0, 0.28, 0.02), rough=0.5)
m_pants = mat("pants", (0.10, 0.12, 0.20), rough=0.7)
m_skin = mat("skin", (0.65, 0.45, 0.32), rough=0.6)
m_helmet = mat("helmet", (1.0, 0.35, 0.05), rough=0.3)

# ---------- floor & wall ----------
box("floor", (30, 20, 0.1), (0, 5, -0.05), m_floor)
box("floor_apron", (30, 1.2, 0.102), (0, 1.05, -0.05), m_floor2)
box("wall", (30, 0.3, 6), (0, -0.15, 3), m_wall)
box("wall_stripe", (30, 0.02, 0.45), (0, 0.012, 2.85), m_blue)
for i in range(-14, 15):
    box(f"seam{i}", (0.03, 0.02, 6), (i * 1.0, 0.011, 3), m_seam)

# ---------- doorway A (forklift, screen center-left => +X) ----------
DX = 0.9
dw, dh = 2.9, 3.1
box("openA", (dw, 0.55, dh), (DX, -0.18, dh / 2), m_dark)
box("lintelA", (dw + 0.5, 0.06, 0.32), (DX, 0.032, dh + 0.16), m_frame)
for i in range(6):
    box(f"slatA{i}", (dw + 0.3, 0.07, 0.085), (DX, 0.05, dh + 0.36 + i * 0.095), m_slat)
box("jambL", (0.14, 0.12, dh), (DX - dw / 2 - 0.08, 0.035, dh / 2), m_frame)
box("jambR", (0.14, 0.12, dh), (DX + dw / 2 + 0.08, 0.035, dh / 2), m_frame)

# ---------- doorway B with boxes (screen right => -X) ----------
BX = -3.3
box("openB", (2.3, 0.55, 2.5), (BX, -0.18, 1.25), m_dark)
box("lintelB", (2.8, 0.06, 0.3), (BX, 0.032, 2.65), m_frame)
for i, (w, d, h) in enumerate([(0.8, 0.55, 0.5), (0.7, 0.5, 0.45), (0.6, 0.45, 0.4)]):
    box(f"boxin{i}", (w, d, h), (BX + 0.1, 0.12, 0.25 + i * 0.47), m_card if i % 2 else m_card2)

# pallet with boxes (foreground right)
PX, PY = -3.9, 3.7
box("pallet", (1.35, 1.1, 0.14), (PX, PY, 0.07), m_pallet)
import random
random.seed(7)
for i in range(2):
    for j in range(2):
        for k in range(2):
            w = 0.55 + random.uniform(-0.04, 0.04)
            box(f"pbox{i}{j}{k}", (w, 0.5, 0.42),
                (PX - 0.31 + i * 0.63, PY - 0.26 + j * 0.53, 0.35 + k * 0.44),
                m_card if (i + j + k) % 2 else m_card2)

# ---------- hazard line ----------
for i in range(-15, 16):
    box(f"hz{i}", (0.5, 0.15, 0.104), (i * 0.5, 4.3, -0.048), m_yellow if i % 2 else m_black)

# ---------- generic autonomous pallet forklift ----------
FX, FY = DX, 1.0
box("fk_body", (0.85, 0.7, 1.05), (FX, FY, 0.62), m_red)
box("fk_base", (0.95, 0.85, 0.25), (FX, FY + 0.05, 0.13), m_darkm)
box("fk_mast1", (0.10, 0.10, 1.9), (FX - 0.18, FY - 0.25, 1.6), m_grey)
box("fk_mast2", (0.10, 0.10, 1.9), (FX + 0.18, FY - 0.25, 1.6), m_grey)
box("fk_cross", (0.5, 0.08, 0.1), (FX, FY - 0.25, 2.45), m_grey)
box("fk_fork1", (0.16, 1.0, 0.06), (FX - 0.22, FY + 0.75, 0.06), m_darkm)
box("fk_fork2", (0.16, 1.0, 0.06), (FX + 0.22, FY + 0.75, 0.06), m_darkm)
box("fk_head", (0.6, 0.35, 0.22), (FX, FY, 1.28), m_sensor)
sphere("fk_cam1", 0.045, (FX - 0.15, FY + 0.19, 1.28), m_screen)
sphere("fk_cam2", 0.045, (FX + 0.15, FY + 0.19, 1.28), m_screen)
box("fk_panel", (0.55, 0.04, 0.35), (FX, FY + 0.37, 0.75), m_darkm)
box("fk_screen", (0.42, 0.02, 0.24), (FX, FY + 0.40, 0.75), m_screen)
cyl("fk_wheel1", 0.13, 0.1, (FX - 0.38, FY + 0.1, 0.13), m_sensor, rot=(0, math.pi / 2, 0))
cyl("fk_wheel2", 0.13, 0.1, (FX + 0.38, FY + 0.1, 0.13), m_sensor, rot=(0, math.pi / 2, 0))
cyl("fk_beacon", 0.05, 0.08, (FX, FY - 0.05, 2.55), m_helmet)

# ---------- worker (orange vest, animated; enters from screen right = -X) ----------
ZONE3D = (DX, 2.4)
bpy.ops.object.empty_add(location=(-9, 2.7, 0))
root = bpy.context.object
root.name = "worker_root"

def wpart(o):
    o.parent = root
    o.location = (o.location.x + 9, o.location.y - 2.7, o.location.z)

wpart(cyl("w_leg1", 0.07, 0.5, (-9 - 0.09, 2.7, 0.25), m_pants))
wpart(cyl("w_leg2", 0.07, 0.5, (-9 + 0.09, 2.7, 0.25), m_pants))
wpart(box("w_torso", (0.40, 0.24, 0.55), (-9, 2.7, 0.85), m_vest))
wpart(cyl("w_arm1", 0.05, 0.45, (-9 - 0.25, 2.7, 0.85), m_vest))
wpart(cyl("w_arm2", 0.05, 0.45, (-9 + 0.25, 2.7, 0.85), m_vest))
wpart(sphere("w_head", 0.11, (-9, 2.7, 1.30), m_skin))
wpart(sphere("w_helmet", 0.12, (-9, 2.7, 1.36), m_helmet, scale=(1, 1, 0.7)))

def key(fr, x, y):
    root.location = (x, y, 0)
    root.keyframe_insert("location", frame=fr)

key(1, -8.5, 2.7)
key(int(5.0 * FPS), -8.5, 2.7)
key(int(9.5 * FPS), -1.4, 2.6)                     # near zone edge
key(int(10.7 * FPS), ZONE3D[0] - 0.55, ZONE3D[1])   # inside zone
key(int(12.0 * FPS), ZONE3D[0] - 0.5, ZONE3D[1])
key(N_FRAMES, -8.5, 2.75)
for fc in root.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

def yaw(fr, deg):
    root.rotation_euler = (0, 0, math.radians(deg))
    root.keyframe_insert("rotation_euler", frame=fr)

yaw(1, 90)
yaw(int(12.0 * FPS), 90)
yaw(int(12.5 * FPS), -90)

# ---------- lights & world ----------
scn.world = bpy.data.worlds.new("world")
scn.world.use_nodes = True
scn.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.37, 0.40, 1)
scn.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.28

def arealight(name, loc, power, size=4):
    bpy.ops.object.light_add(type="AREA", location=loc)
    L = bpy.context.object
    L.name = name
    L.data.energy = power
    L.data.size = size
    return L

arealight("key", (0, 5.5, 6.5), 1800, 7)
arealight("fill", (-5, 8, 5), 450, 4)
L = arealight("door_glow", (DX, 0.4, 1.5), 60, 2)
L.data.color = (0.7, 0.8, 1.0)
W = arealight("wall_wash", (0, 7.0, 4.8), 1400, 8)   # aimed at the wall
W.rotation_euler = (math.radians(-75), 0, 0)

# ---------- camera ----------
bpy.ops.object.camera_add(location=(-0.7, 8.8, 3.9))
cam = bpy.context.object
cam.data.lens = 34
bpy.ops.object.empty_add(location=(0.25, 1.7, 0.75))
target = bpy.context.object
target.name = "cam_target"
tc = cam.constraints.new("TRACK_TO")
tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"
scn.camera = cam

# print zone screen-space coords for src/config.py
import bpy_extras
from mathutils import Vector
deps = bpy.context.evaluated_depsgraph_get()
scn.frame_set(1)
c = bpy_extras.object_utils.world_to_camera_view(scn, cam, Vector((ZONE3D[0], ZONE3D[1], 0)))
e1 = bpy_extras.object_utils.world_to_camera_view(scn, cam, Vector((ZONE3D[0] + 1.1, ZONE3D[1], 0)))
e2 = bpy_extras.object_utils.world_to_camera_view(scn, cam, Vector((ZONE3D[0], ZONE3D[1] + 0.75, 0)))
print(f"ZONE_SCREEN center=({c.x:.3f},{1-c.y:.3f}) rx={abs(e1.x-c.x):.3f} ry={abs(e2.y-c.y):.3f}")

# ---------- CLI ----------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if argv and argv[0] == "--still":
    scn.frame_set(int(argv[1]))
    scn.render.filepath = argv[2]
    bpy.ops.render.render(write_still=True)
elif argv and argv[0] == "--hero":
    # --hero frame out.png y0 y1  (960x540, 28 samples, horizontal border strip)
    scn.render.resolution_x, scn.render.resolution_y = 960, 540
    scn.cycles.samples = 28
    scn.render.use_border = True
    scn.render.use_crop_to_border = False
    scn.render.border_min_x, scn.render.border_max_x = 0.0, 1.0
    scn.render.border_min_y, scn.render.border_max_y = float(argv[3]), float(argv[4])
    scn.frame_set(int(argv[1]))
    scn.render.filepath = argv[2]
    bpy.ops.render.render(write_still=True)
elif argv and argv[0] == "--anim":
    # --anim out_dir start end step [budget_s] ; skips existing frames, stops on budget
    import os, time
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
