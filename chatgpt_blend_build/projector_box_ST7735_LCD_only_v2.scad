// DIY Mini Projector Box - ST7735S 1.8" LCD-only holder, v2
// Units: mm
$fn = 120;
part = "body";
L=145; W=100; H=92; wall=3.0; front_t=9.0; rear_t=4.0;
lens_d=80.0; lens_fit_d=80.8; lens_pocket_depth=5.5; lens_aperture_d=74.0; lens_cy=W/2; lens_cz=H/2;
lcd_glass_y=46.7; lcd_glass_z=34.7; lcd_glass_t=2.3;
lcd_active_y=35.04; lcd_active_z=28.03;
lcd_xy_clear=0.60; lcd_t_clear=0.50;
lcd_pocket_y=lcd_glass_y+lcd_xy_clear; lcd_pocket_z=lcd_glass_z+lcd_xy_clear; lcd_pocket_x=lcd_glass_t+lcd_t_clear;
lcd_holder_x=4.8; lcd_holder_y=59.0; lcd_holder_z=47.0;
lcd_window_y=37.0; lcd_window_z=30.0; fpc_relief_z=18.0; fpc_relief_x=3.4;
lcd_positions=[50,60,70,80,90]; lcd_preview_x=70; slot_fit=lcd_holder_x+0.55; rib_x=1.8; rail_touch=2.0;
lcd_bottom_z=lens_cz-lcd_holder_z/2;
led_window=40.0; led_mount_spacing=52.0; led_screw_d=3.4; led_center_y=lens_cy; led_center_z=lens_cz; cable_d=7.0;
lid_t=3.0; lid_lip_h=2.0; lid_clearance=0.45;

module lcd_guide_set(slotx){
 holder_y0=(W-lcd_holder_y)/2; holder_y1=holder_y0+lcd_holder_y;
 for(xpos=[slotx-slot_fit/2-rib_x,slotx+slot_fit/2]){
  translate([xpos,wall,lcd_bottom_z-1.0]) cube([rib_x,holder_y0-wall+rail_touch,lcd_holder_z+2.0]);
  translate([xpos,holder_y1-rail_touch,lcd_bottom_z-1.0]) cube([rib_x,W-wall-(holder_y1-rail_touch),lcd_holder_z+2.0]);
 }
 translate([slotx-slot_fit/2-rib_x,holder_y0,lcd_bottom_z-2.0]) cube([2*rib_x+slot_fit,lcd_holder_y,2.0]);
}
module shell_raw(){
 union(){
  cube([L,W,wall]); cube([L,wall,H]); translate([0,W-wall,0]) cube([L,wall,H]);
  cube([front_t,W,H]); translate([L-rear_t,0,0]) cube([rear_t,W,H]);
  for(slotx=lcd_positions) lcd_guide_set(slotx);
 }
}
module body(){
 difference(){
  shell_raw();
  translate([-0.2,lens_cy,lens_cz]) rotate([0,90,0]) cylinder(h=lens_pocket_depth+0.4,d=lens_fit_d);
  translate([lens_pocket_depth-0.1,lens_cy,lens_cz]) rotate([0,90,0]) cylinder(h=front_t-lens_pocket_depth+0.5,d=lens_aperture_d);
  translate([L-rear_t-0.2,led_center_y-led_window/2,led_center_z-led_window/2]) cube([rear_t+0.5,led_window,led_window]);
  for(yy=[-led_mount_spacing/2,led_mount_spacing/2]) for(zz=[-led_mount_spacing/2,led_mount_spacing/2])
   translate([L-rear_t-0.2,led_center_y+yy,led_center_z+zz]) rotate([0,90,0]) cylinder(h=rear_t+0.5,d=led_screw_d);
  translate([L-rear_t-0.2,W-15,15]) rotate([0,90,0]) cylinder(h=rear_t+0.5,d=cable_d);
  for(xx=[105:10:130]) for(zz=[26,39,52,65]){
   translate([xx,-0.2,zz]) cube([6,wall+0.5,3.0]);
   translate([xx,W-wall-0.2,zz]) cube([6,wall+0.5,3.0]);
  }
 }
}
module lcd_holder(){
 difference(){
  cube([lcd_holder_x,lcd_holder_y,lcd_holder_z]);
  translate([-0.2,(lcd_holder_y-lcd_window_y)/2,(lcd_holder_z-lcd_window_z)/2]) cube([lcd_holder_x+0.4,lcd_window_y,lcd_window_z]);
  translate([lcd_holder_x-lcd_pocket_x,(lcd_holder_y-lcd_pocket_y)/2,(lcd_holder_z-lcd_pocket_z)/2]) cube([lcd_pocket_x+0.2,lcd_pocket_y,lcd_pocket_z]);
  translate([lcd_holder_x-fpc_relief_x,-0.2,(lcd_holder_z-fpc_relief_z)/2]) cube([fpc_relief_x+0.3,(lcd_holder_y-lcd_pocket_y)/2+1.2,fpc_relief_z]);
  translate([lcd_holder_x-fpc_relief_x,lcd_holder_y-(lcd_holder_y-lcd_pocket_y)/2-1.0,(lcd_holder_z-fpc_relief_z)/2]) cube([fpc_relief_x+0.3,(lcd_holder_y-lcd_pocket_y)/2+1.2,fpc_relief_z]);
 }
}
module lens_retainer(){difference(){cylinder(h=2.5,d=lens_d+0.25); translate([0,0,-0.1]) cylinder(h=2.7,d=lens_aperture_d);}}
module lid(){
 difference(){
  union(){
   cube([L,W,lid_t]);
   translate([front_t+lid_clearance,wall+lid_clearance,lid_t]) cube([L-front_t-rear_t-2*lid_clearance,2.0,lid_lip_h]);
   translate([front_t+lid_clearance,W-wall-lid_clearance-2.0,lid_t]) cube([L-front_t-rear_t-2*lid_clearance,2.0,lid_lip_h]);
   translate([front_t+lid_clearance,wall+lid_clearance,lid_t]) cube([2.0,W-2*wall-2*lid_clearance,lid_lip_h]);
   translate([L-rear_t-lid_clearance-2.0,wall+lid_clearance,lid_t]) cube([2.0,W-2*wall-2*lid_clearance,lid_lip_h]);
  }
  translate([57,-0.2,-0.2]) cube([28,wall+8,5.5]);
 }
}
if(part=="body") body();
else if(part=="lid") lid();
else if(part=="lcd_holder") lcd_holder();
else if(part=="lens_retainer") lens_retainer();
