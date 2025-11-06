import bpy
import sys
import math
import random
import os
import mathutils

# Add the lib directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from dataset_utils import get_dataset_dict

"""
command format:
blender --background --python render_multiview.py -- input_path output_dir
blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1011/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1011/multiview

where input_path is the directory you store the .obj files
output_dir is the place you store multiview images
"""

argv = sys.argv
argv = argv[argv.index("--") + 1:]
class_name = argv[0]

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
                # newer property that controls which denoiser to use
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
    # mix.inputs[1] is Shader socket 1, mix.inputs[2] is Shader socket 2
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    # Set blend mode for transparency (important if using Eevee; for Cycles it's fine)
    # Ensure material has correct settings for transparency in Eevee
    mat.blend_method = 'BLEND'
    #mat.shadow_method = 'NONE'  # ensure it doesn't cast shadow in Eevee
    return mat

# ---------------------
# Start script actions
# ---------------------

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
# For newer Blender versions, also set device preferences
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'  # Or 'OPTIX' for RTX cards
prefs.get_devices()
bpy.context.scene.cycles.device = 'GPU'

print(f"blender device is:{bpy.context.scene.cycles.device}")



id_dict = get_dataset_dict()
assert class_name in id_dict, f"Not Exist CLASS {class_name}"

for id in id_dict[class_name][15:]:
    print(f"start process id:{id} object")
    output_dir = f"/work/u9497859/shared_data/partnet-mobility-v0/dataset/{id}/multiview"
    input_path = f"/work/u9497859/shared_data/partnet-mobility-v0/dataset/{id}/textured_objs"

    os.makedirs(output_dir, exist_ok=True)


    # Remove default(previous) objects (clear scene)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False) 

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

    """
    # If nothing imported, abort
    if len([o for o in bpy.data.objects if o.type == 'MESH']) == 0:
        raise RuntimeError("No mesh objects found after importing. Aborting.")

    # Join all imported meshes into a single object (so we render combined obj)
    mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    if len(mesh_objs) > 1:
        # Select all mesh objects and join
        for o in mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        try:
            bpy.ops.object.join()
        except Exception as e:
            print("Warning: join failed:", e)
    # After join, get the single mesh object (either joined or only one)
    all_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    obj = all_meshes[0]
    """


    # Center object
    #for obj in bpy.context.selected_objects:
    #   obj.location = (0, 0, 0)
    # test:

    for obj in bpy.context.selected_objects:
        obj.location = (0, 0, 0)

    # Add camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    # (Optional) Add area light - you can keep or remove; camera-only material will prevent shadows
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
    area_light = bpy.context.object
    area_light.data.energy = 5000
    area_light.data.size = 30

    # Use Cycles and set transparent film
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    # Transparent background (useful for CV datasets)
    try:
        scene.render.film_transparent = True
    except Exception:
        pass

    # Disable denoising / compositor to avoid "Build without OpenImageDenoiser" errors
    disable_denoising_and_compositor(scene)

    
    # Create and assign the camera-only material (constant color)
    cam_only_mat = create_camera_only_material(name="CameraOnly_CamRay", color=(0.8, 0.8, 0.8, 1.0))

    # Only apply materials to mesh objects
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH' and hasattr(obj.data, 'materials'):
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