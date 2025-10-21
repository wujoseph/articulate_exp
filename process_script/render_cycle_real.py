import bpy
import sys
import math
import random
import os
import mathutils

"""
command format:
blender --background --python render_multiview.py -- input_path output_dir

where input_path is the directory you store the .obj files
output_dir is the place you store multiview images
"""

argv = sys.argv
argv = argv[argv.index("--") + 1:]
input_path = argv[0]
output_dir = argv[1]

# --- Helper: disable denoising / compositor to avoid OIDN errors ---
def disable_denoising_and_compositor(scene):
    # Disable Cycles denoising on view layers and scene (if present)
    try:
        if scene.render.engine == 'CYCLES':
            for vl in scene.view_layers:
                if hasattr(vl, "cycles") and hasattr(vl.cycles, "use_denoising"):
                    vl.cycles.use_denoising = False
            if hasattr(scene, "cycles"):
                if hasattr(scene.cycles, "use_denoising"):
                    scene.cycles.use_denoising = False
                if hasattr(scene.cycles, "denoiser"):
                    try:
                        scene.cycles.denoiser = 'NONE'
                    except Exception:
                        pass
    except Exception as e:
        print("Warning disabling cycles denoising:", e)

    # Remove compositor Denoise nodes or disable compositor nodes
    try:
        if getattr(scene, "use_nodes", False) and scene.node_tree:
            nodes = scene.node_tree.nodes
            deno_nodes = [n for n in nodes if getattr(n, "type", "") == "DENOISE" or getattr(n, "bl_idname", "") == "CompositorNodeDenoise"]
            for n in deno_nodes:
                print("Removing compositor denoise node:", n)
                nodes.remove(n)
            scene.use_nodes = False
    except Exception as e:
        print("Warning modifying compositor nodes:", e)

# --- Helper: create camera-only material (camera rays show color, other rays transparent) ---
def create_camera_only_material(name="CameraOnlyMat", color=(0.8, 0.8, 0.8, 1.0)):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Nodes
    tex_color = nodes.new(type='ShaderNodeRGB')
    tex_color.outputs[0].default_value = color
    tex_color.location = (-400, 0)

    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (-150, 0)
    links.new(tex_color.outputs['Color'], emission.inputs['Color'])

    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    transparent.location = (-150, -200)

    mix = nodes.new(type='ShaderNodeMixShader')
    mix.location = (100, 0)

    lightpath = nodes.new(type='ShaderNodeLightPath')
    lightpath.location = (-400, 200)

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)

    # Use Is Camera Ray as factor: camera rays see emission, other rays see transparent (no shadows)
    links.new(lightpath.outputs['Is Camera Ray'], mix.inputs['Fac'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    # For Cycles, Opaque mode is fine; keep shadow_method none just in case
    mat.blend_method = 'OPAQUE'
    try:
        mat.shadow_method = 'NONE'
    except Exception:
        pass
    return mat

# ---------------------
# Start script actions
# ---------------------

# Remove default objects (clear scene)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Make sure output dir exists
os.makedirs(output_dir, exist_ok=True)


if not os.path.isdir(input_path):
    print(f"Error: Directory not found - {input_path}")
else:
    # Iterate through all files in the specified directory
    for filename in os.listdir(input_path):
        # Check if the file is an OBJ file
        if filename.lower().endswith('.obj'):
            filepath = os.path.join(input_path, filename)
            print(f"Importing: {filepath}")
            
            try:
                # Use the import_scene.obj operator to import the OBJ file
                # You can customize import options like 'forward_axis' and 'up_axis'
                bpy.ops.wm.obj_import(filepath=filepath)
                print(f"Successfully imported: {filename}")
            except Exception as e:
                print(f"Error importing {filename}: {e}")

# Import GLB
#bpy.ops.import_scene.gltf(filepath=input_path)

# Center object
for obj in bpy.context.selected_objects:
    obj.location = (0, 0, 0)


# Add camera
bpy.ops.object.camera_add()
camera = bpy.context.object
bpy.context.scene.camera = camera

# (Optional) Add area light - kept for completeness (camera-only material ignores lighting for camera rays)
bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
area_light = bpy.context.object
area_light.data.energy = 5000
area_light.data.size = 30

# Use Cycles
scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# Set a BLACK world background and avoid transparent film for faster renders
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
# get/create Background node
bg_node = None
try:
    bg_node = scene.world.node_tree.nodes.get('Background')
except Exception:
    bg_node = None

if bg_node is None:
    # create background node and output if necessary
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    nodes.clear()
    bg_node = nodes.new(type='ShaderNodeBackground')
    out_node = nodes.new(type='ShaderNodeOutputWorld')
    links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])

# Set background color to black and reasonably low strength for speed
bg_node.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # black RGBA
bg_node.inputs[1].default_value = 1.0  # strength

# Ensure film transparent is disabled (faster than generating alpha)
try:
    scene.render.film_transparent = False
except Exception:
    pass

# Disable denoising / compositor to avoid "Build without OpenImageDenoiser" errors
disable_denoising_and_compositor(scene)

# Create and assign the camera-only material (constant color)
cam_only_mat = create_camera_only_material(name="CameraOnly_CamRay", color=(0.8, 0.8, 0.8, 1.0))
# Assign to object (replace materials)
if obj.data.materials:
    obj.data.materials.clear()
obj.data.materials.append(cam_only_mat)

# Rendering settings
scene.render.resolution_x = 512
scene.render.resolution_y = 512

# Multi-view rendering (random spherical sampling)
num_views = 10
radius = 2.5

for i in range(num_views):
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(20), math.radians(160))
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    camera.location = (x, y, z)

    target = mathutils.Vector((0, 0, 0))
    cam_location = camera.location
    direction = target - cam_location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()

    scene.render.filepath = os.path.join(output_dir, f"view_{i:02d}.png")
    bpy.ops.render.render(write_still=True)

print("Done. Images saved to:", output_dir)