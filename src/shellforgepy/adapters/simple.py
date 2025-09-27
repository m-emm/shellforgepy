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
directed_cylinder_at = adapter.directed_cylinder_at
fuse_parts = adapter.fuse_parts
get_bounding_box = adapter.get_bounding_box
translate_part = adapter.translate_part
rotate_part = adapter.rotate_part
export_solid_to_stl = adapter.export_solid_to_stl
copy_part = adapter.copy_part

__all__ = [
    "create_basic_box",
    "create_basic_cylinder",
    "create_basic_sphere",
    "create_solid_from_traditional_face_vertex_maps",
    "create_basic_cone",
    "create_text_object",
    "directed_cylinder_at",
    "get_bounding_box",
    "fuse_parts",
    "translate_part",
    "rotate_part",
    "export_solid_to_stl",
    "copy_part",
]
