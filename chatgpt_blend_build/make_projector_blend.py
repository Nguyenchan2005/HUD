import bpy, math, subprocess, os
from mathutils import Vector
BASE=os.getcwd()
SCAD=os.path.join(BASE,'projector_box_ST7735_LCD_only_v2.scad')
def run_scad(part,out):
    txt=open(SCAD,'r',encoding='utf-8').read()
    txt=txt.replace('part = "body";', f'part = "{part}";')
    tmp=os.path.join(BASE,f'_{part}.scad')
    open(tmp,'w',encoding='utf-8').write(txt)
    subprocess.check_call(['openscad','-o',out,tmp])
def mat(name,c,a=1.0):
    m=bpy.data.materials.new(name); m.diffuse_color=(*c,a); m.use_nodes=True
    bsdf=m.node_tree.nodes.get('Principled BSDF'); bsdf.inputs['Base Color'].default_value=(*c,1); bsdf.inputs['Alpha'].default_value=a; bsdf.inputs['Roughness'].default_value=.45
    if a<1:
        m.blend_method='BLEND'
    return m
def import_stl(path,name,matl,loc=(0,0,0),rot=(0,0,0)):
    bpy.ops.import_mesh.stl(filepath=path)
    o=bpy.context.active_object; o.name=name; o.location=loc; o.rotation_euler=rot; o.data.materials.append(matl); return o
for o in list(bpy.data.objects): bpy.data.objects.remove(o,do_unlink=True)
# build STLs
parts={'body':'body.stl','lid':'lid.stl','lcd_holder':'lcd_holder.stl','lens_retainer':'lens_retainer.stl'}
for p,f in parts.items(): run_scad(p,os.path.join(BASE,f))
mb=mat('Body',(0.65,0.67,0.7)); ml=mat('Lid',(0.82,0.84,0.87),.82); mh=mat('LCD holder',(0.12,0.35,0.75)); mr=mat('Lens retainer',(0.15,0.16,0.18))
import_stl(os.path.join(BASE,'body.stl'),'01_BODY',mb)
import_stl(os.path.join(BASE,'lid.stl'),'02_LID_exploded',ml,(0,0,107))
import_stl(os.path.join(BASE,'lcd_holder.stl'),'03_LCD_GLASS_HOLDER',mh,(67.6,20.5,22.5))
import_stl(os.path.join(BASE,'lens_retainer.stl'),'04_LENS_RETAINER',mr,(-2.5,50,46),(0,math.radians(90),0))
# LCD glass only
mglass=mat('LCD glass',(0.02,0.09,0.11),.58); mact=mat('Active area',(0.03,0.75,0.85),.48); mfpc=mat('FPC',(0.72,0.28,0.04),.95)
def cube(name,dims,center,matl):
    bpy.ops.mesh.primitive_cube_add(size=1,location=center)
    o=bpy.context.active_object; o.name=name; o.dimensions=dims; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); o.data.materials.append(matl); return o
lcd=cube('LCD_GLASS_ONLY_46.7x34.7x2.3mm',(2.3,46.7,34.7),(71,50,46),mglass)
active=cube('LCD_ACTIVE_AREA_35.04x28.03mm',(0.18,35.04,28.03),(69.76,50,46),mact)
cube('LCD_FPC_REFERENCE',(0.12,9,17),(70,76.7,46),mfpc)
scene=bpy.context.scene
scene['NOTE']='LCD glass only; no PCB/controller board.'
scene['LCD_ACTIVE_AREA_MM']='35.04 x 28.03'
scene['LCD_GLASS_REFERENCE_MM']='46.7 x 34.7 x 2.3'
scene.unit_settings.system='METRIC'; scene.unit_settings.length_unit='MILLIMETERS'; scene.unit_settings.scale_length=.001
# camera
bpy.ops.object.camera_add(location=(245,-205,185)); cam=bpy.context.active_object
cam.rotation_euler=((Vector((70,50,48))-cam.location).to_track_quat('-Z','Y').to_euler()); cam.data.lens=52; scene.camera=cam
bpy.ops.object.light_add(type='AREA',location=(100,-70,180)); key=bpy.context.active_object; key.data.energy=1200; key.data.size=120
key.rotation_euler=((Vector((70,50,45))-key.location).to_track_quat('-Z','Y').to_euler())
scene.render.engine='BLENDER_EEVEE'; scene.render.resolution_x=1000; scene.render.resolution_y=700; scene.render.resolution_percentage=100
scene.render.filepath=os.path.join(BASE,'projector_box_ST7735_LCD_only_v2_assembly_preview.png')
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(BASE,'projector_box_ST7735_LCD_only_v2_assembly.blend'))
bpy.ops.render.render(write_still=True)
