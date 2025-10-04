from httpx import get
from shellforgepy.adapters.adapter_chooser import get_cad_adapter

selected_adapter = get_cad_adapter()


create_box = selected_adapter.create_box
create_cylinder = selected_adapter.create_cylinder
create_sphere = selected_adapter.create_sphere
create_solid_from_traditional_face_vertex_maps = (
    selected_adapter.create_solid_from_traditional_face_vertex_maps
)
create_cone = selected_adapter.create_cone
create_text_object = selected_adapter.create_text_object
fuse_parts = selected_adapter.fuse_parts
cut_parts = selected_adapter.cut_parts
create_extruded_polygon = selected_adapter.create_extruded_polygon
get_volume = selected_adapter.get_volume

get_bounding_box = selected_adapter.get_bounding_box
translate_part = selected_adapter.translate_part
rotate_part = selected_adapter.rotate_part
export_solid_to_stl = selected_adapter.export_solid_to_stl
copy_part = selected_adapter.copy_part
create_filleted_box = selected_adapter.create_filleted_box
translate_part_native = selected_adapter.translate_part_native
rotate_part_native = selected_adapter.rotate_part_native
apply_fillet_to_edges = selected_adapter.apply_fillet_to_edges
apply_fillet_by_alignment = selected_adapter.apply_fillet_by_alignment
get_adapter_id = selected_adapter.get_adapter_id
mirror_part = selected_adapter.mirror_part
get_bounding_box_center = selected_adapter.get_bounding_box_center
get_bounding_box_size = selected_adapter.get_bounding_box_size
get_vertices = selected_adapter.get_vertices
create_extruded_polygon = selected_adapter.create_extruded_polygon
copy_part = selected_adapter.copy_part
create_hex_prism = selected_adapter.create_hex_prism
get_vertex_coordinates = selected_adapter.get_vertex_coordinates
get_vertex_coordinates_np = selected_adapter.get_vertex_coordinates_np
__all__ = [
    "apply_fillet_by_alignment",
    "apply_fillet_to_edges",
    "copy_part",
    "create_box",
    "create_cone",
    "create_cylinder",
    "create_extruded_polygon",
    "create_filleted_box",
    "create_hex_prism",
    "create_solid_from_traditional_face_vertex_maps",
    "create_sphere",
    "create_text_object",
    "cut_parts",
    "export_solid_to_stl",
    "fuse_parts",
    "get_adapter_id",
    "get_bounding_box_center",
    "get_bounding_box_size",
    "get_bounding_box",
    "get_vertex_coordinates_np",
    "get_vertex_coordinates",
    "get_vertices",
    "get_volume",
    "mirror_part",
    "rotate_part_native",
    "rotate_part",
    "translate_part_native",
    "translate_part",
]
