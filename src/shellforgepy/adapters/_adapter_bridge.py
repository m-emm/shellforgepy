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
scale_part = selected_adapter.scale_part
export_solid_to_stl = selected_adapter.export_solid_to_stl
export_solid_to_step = selected_adapter.export_solid_to_step
export_solid_to_obj = selected_adapter.export_solid_to_obj
export_colored_parts_to_obj = selected_adapter.export_colored_parts_to_obj
export_structured_step = selected_adapter.export_structured_step
import_solid_from_step = selected_adapter.import_solid_from_step
deserialize_structured_step = selected_adapter.deserialize_structured_step
copy_part = selected_adapter.copy_part
create_filleted_box = selected_adapter.create_filleted_box
translate_part_native = selected_adapter.translate_part_native
rotate_part_native = selected_adapter.rotate_part_native
mirror_part_native = selected_adapter.mirror_part_native
scale_part_native = selected_adapter.scale_part_native
apply_fillet_to_edges = selected_adapter.apply_fillet_to_edges
filter_edges_by_function = selected_adapter.filter_edges_by_function
apply_fillet_by_alignment = selected_adapter.apply_fillet_by_alignment
get_adapter_id = selected_adapter.get_adapter_id
mirror_part = selected_adapter.mirror_part
get_bounding_box_center = selected_adapter.get_bounding_box_center
get_bounding_box_size = selected_adapter.get_bounding_box_size
get_vertices = selected_adapter.get_vertices
create_extruded_polygon = selected_adapter.create_extruded_polygon
copy_part = selected_adapter.copy_part
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
    "create_solid_from_traditional_face_vertex_maps",
    "create_sphere",
    "create_text_object",
    "cut_parts",
    "deserialize_structured_step",
    "export_colored_parts_to_obj",
    "export_solid_to_obj",
    "export_solid_to_stl",
    "export_solid_to_step",
    "export_structured_step",
    "fuse_parts",
    "filter_edges_by_function",
    "get_adapter_id",
    "get_bounding_box_center",
    "get_bounding_box_size",
    "get_bounding_box",
    "get_vertex_coordinates_np",
    "get_vertex_coordinates",
    "get_vertices",
    "get_volume",
    "import_solid_from_step",
    "mirror_part",
    "mirror_part_native",
    "rotate_part_native",
    "rotate_part",
    "scale_part_native",
    "scale_part",
    "translate_part_native",
    "translate_part",
]
