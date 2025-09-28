from shellforgepy.adapters.adapter_chooser import get_cad_adapter

adapter = get_cad_adapter()


create_basic_box = adapter.create_basic_box
create_basic_cylinder = adapter.create_basic_cylinder
create_basic_sphere = adapter.create_basic_sphere
create_solid_from_traditional_face_vertex_maps = (
    adapter.create_solid_from_traditional_face_vertex_maps
)
create_basic_cone = adapter.create_basic_cone
create_text_object = adapter.create_text_object
fuse_parts = adapter.fuse_parts
cut_parts = adapter.cut_parts
create_extruded_polygon = adapter.create_extruded_polygon
get_volume = adapter.get_volume

get_bounding_box = adapter.get_bounding_box
translate_part = adapter.translate_part
rotate_part = adapter.rotate_part
export_solid_to_stl = adapter.export_solid_to_stl
copy_part = adapter.copy_part
create_filleted_box = adapter.create_filleted_box
translate_part_native = adapter.translate_part_native
rotate_part_native = adapter.rotate_part_native

__all__ = [
    "create_basic_box",
    "create_basic_cylinder",
    "create_basic_sphere",
    "create_solid_from_traditional_face_vertex_maps",
    "create_basic_cone",
    "create_text_object",
    "get_bounding_box",
    "fuse_parts",
    "cut_parts",
    "translate_part",
    "rotate_part",
    "export_solid_to_stl",
    "copy_part",
    "create_extruded_polygon",
    "get_volume",
    "create_filleted_box",
    "translate_part_native",
    "rotate_part_native",
]
