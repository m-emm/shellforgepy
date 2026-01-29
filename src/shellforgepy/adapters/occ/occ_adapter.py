"""Thin wrapper that delegates to the installed shellforgepy-occ-adapter."""

from shellforgepy_occ_adapter import occ_adapter as _oa

# Re-export everything the core adapter provides so it matches CadQuery API
get_adapter_id = _oa.get_adapter_id
create_box = _oa.create_box
create_cylinder = _oa.create_cylinder
create_sphere = _oa.create_sphere
create_cone = _oa.create_cone
create_solid_from_traditional_face_vertex_maps = (
    _oa.create_solid_from_traditional_face_vertex_maps
)
create_text_object = getattr(_oa, "create_text_object", None)
create_extruded_polygon = _oa.create_extruded_polygon
create_filleted_box = _oa.create_filleted_box
fuse_parts = _oa.fuse_parts
cut_parts = _oa.cut_parts
get_volume = _oa.get_volume
get_bounding_box = _oa.get_bounding_box
get_bounding_box_center = _oa.get_bounding_box_center
get_bounding_box_size = _oa.get_bounding_box_size
get_bounding_box_min = _oa.get_bounding_box_min
get_bounding_box_max = _oa.get_bounding_box_max
get_z_min = _oa.get_z_min
get_z_max = _oa.get_z_max
get_bounding_box_center_np = _oa.get_bounding_box_center_np
get_bounding_box_min_np = _oa.get_bounding_box_min_np
get_bounding_box_max_np = _oa.get_bounding_box_max_np
get_bounding_box_size_np = _oa.get_bounding_box_size_np
get_vertices = getattr(_oa, "get_vertices", None)
get_vertex_coordinates = getattr(_oa, "get_vertex_coordinates", None)
get_vertex_coordinates_np = getattr(_oa, "get_vertex_coordinates_np", None)
translate_part = _oa.translate_part
rotate_part = _oa.rotate_part
scale_part = _oa.scale_part
mirror_part = getattr(_oa, "mirror_part", None)
translate_part_native = getattr(_oa, "translate_part_native", _oa.translate_part)
rotate_part_native = getattr(_oa, "rotate_part_native", _oa.rotate_part)
scale_part_native = getattr(_oa, "scale_part_native", _oa.scale_part)
tessellate_part_native = getattr(_oa, "tessellate_part_native", _oa.tesellate)
mirror_part_native = getattr(_oa, "mirror_part_native", mirror_part)
apply_fillet_to_edges = getattr(_oa, "apply_fillet_to_edges", None)
apply_fillet_by_alignment = getattr(_oa, "apply_fillet_by_alignment", None)
filter_edges_by_function = getattr(_oa, "filter_edges_by_function", None)
copy_part = _oa.copy_part
tesellate = _oa.tesellate
export_solid_to_stl = _oa.export_solid_to_stl
export_solid_to_step = _oa.export_solid_to_step
export_solid_to_obj = getattr(_oa, "export_solid_to_obj", None)
export_colored_parts_to_obj = getattr(_oa, "export_colored_parts_to_obj", None)
export_structured_step = getattr(_oa, "export_structured_step", None)
import_solid_from_step = _oa.import_solid_from_step
deserialize_structured_step = getattr(_oa, "deserialize_structured_step", None)

__all__ = [name for name in globals() if not name.startswith("_")]
