import bpy
import sys
import math
import random
import os
import mathutils


"""
command format:
dont need any conda environment
cd to current directory

blender --background --python render_multiview.py -- input_path output_dir

where input_path is the directory you store the .obj file
output_dir is the place you store multiview

"""


argv = sys.argv
argv = argv[argv.index("--") + 1:]
input_path = argv[0]
output_dir = argv[1]

# Remove default objects
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

# Import GLB
#bpy.ops.import_scene.gltf(filepath=input_path)

# Center object
for obj in bpy.context.selected_objects:
    obj.location = (0, 0, 0)

# Add camera
bpy.ops.object.camera_add()
camera = bpy.context.object
bpy.context.scene.camera = camera

# Add light
"""
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
sun = bpy.context.object
sun.data.energy = 20

bpy.ops.object.light_add(type='SUN', location=(0, 10, 0))
sun = bpy.context.object
sun.data.energy = 20
"""


bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
area_light = bpy.context.object
area_light.data.energy = 5000  # increase for brightness
area_light.data.size = 30     # large size for softness




# set back ground to black
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0, 0, 0, 1)  # RGBA for black
# but not much of strength
bg.inputs[1].default_value = 0  # strength


"""

# set back ground to white
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[0].default_value = (1, 1, 1, 1)  # RGBA for black
# but not much of strength
bg.inputs[1].default_value = 0.1  # strength

"""
# Rendering settings
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Multi-view rendering
num_views = 10
radius = 2.5

for i in range(num_views):
    # Random or systematic angle
    
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
    #camera.look_at = (0, 0, 0)
    bpy.context.view_layer.update()

    """
    # Point camera at origin
    direction = [0 - x, 0 - y, 0 - z]
    rot_quat = camera.rotation_euler.to_quaternion()
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    """

    # Render
    bpy.context.scene.render.filepath = os.path.join(output_dir, f"view_{i:02d}.png")
    bpy.ops.render.render(write_still=True)