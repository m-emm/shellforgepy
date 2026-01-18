"""
Leader-Followers-Cutters Part: CAD-Backend Agnostic Composite Parts

This module implements a composite part system that works with any CAD backend
(CadQuery, FreeCAD, etc.) without backend-specific conditional logic. The design
principles are:

1. Polymorphic Operations: Rely on duck typing and common method signatures
2. Adapter Pattern: Use adapter layer functions for backend-specific operations
3. Flexible Signatures: Use *args/**kwargs to accommodate different parameter formats
4. Reconstruct Pattern: Handle functional transformations through reconstruction
5. No Conditional Logic: Never use "if backend == 'cadquery'" type checks

This approach ensures the composite part system remains maintainable and extensible
as new CAD backends are added to the framework.
"""

import copy
import logging
from types import SimpleNamespace

from shellforgepy.adapters._adapter import (
    copy_part,
    get_bounding_box,
    mirror_part_native,
    rotate_part_native,
    scale_part_native,
    translate_part_native,
)
from shellforgepy.adapters.freecad.freecad_adapter import get_vertex_points
from shellforgepy.construct.named_part import NamedPart
from shellforgepy.construct.part_collector import PartCollector

_logger = logging.getLogger(__name__)


def _ensure_list(items):
    """Convert items to a list, handling None, single items, and existing lists.

    Args:
        items: None, single item, tuple, or list

    Returns:
        List containing the items (empty list if items was None)
    """
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        return [items]
    return list(items)


class LeaderFollowersCuttersPart:
    """A composite CAD part that groups a main leader with associated components.

    This class represents a complete mechanical assembly consisting of a primary part
    and its related components, each serving different roles in the manufacturing and
    assembly process.

    **CAD Backend Agnostic Design:**

    This class is designed to work with any CAD backend (CadQuery, FreeCAD, etc.) without
    requiring backend-specific code. It relies on the adapter pattern where transformation
    and geometric operations are delegated to backend-agnostic helper functions. The class
    never contains conditional logic like "if backend == 'cadquery'" - instead it uses
    polymorphic operations and the reconstruct() pattern for backend compatibility.

    **Part Types:**

    - **leader**: The main part representing the object. This is typically the part
      that gets manufactured (3D printed, machined, etc.). All geometric operations
      (fuse, cut, transform) primarily affect the leader.

    - **followers**: Additional parts associated with the leader for assembly purposes.
      Examples include mounting brackets, clips, or connectors that get "attached" to
      other parts in the assembly. These move with the leader but maintain their
      relative positions.

    - **cutters**: Geometric shapes used to modify other parts to accommodate this
      assembly. Examples include drill holes for screws, slots for tabs, or cutouts
      for proper spacing and fitment. These define negative space in other components.

    - **non_production_parts**: Reference parts not intended for manufacturing but
      useful for visualization, assembly instructions, or design validation. Examples
      include purchased screws, bearings, or standard hardware components.

    **Operations:**

    - **fuse()**: Combines two composite parts. Leaders are geometrically fused while
      all associated parts from both composites are merged. Name collisions raise errors.

    - **cut()**: Removes material from the leader using another part as a cutter. Only
      the original part's followers, cutters, and non-production parts are preserved;
      the cutting part's components are not merged.

    - **use_as_cutter_on()**: Applies all cutters from this composite to a target part,
      effectively "preparing" the target for assembly with this part.

    **Naming System:**

    Components can be optionally named for programmatic access:
    - follower_names: List of names for followers (must match follower count)
    - cutter_names: List of names for cutters (must match cutter count)
    - non_production_names: List of names for non-production parts

    Named components can be retrieved by index using get_*_index_by_name() methods.

    **Example:**
    ```python
    # Create a bracket with mounting holes and reference hardware
    bracket = LeaderFollowersCuttersPart(
        leader=main_bracket_shape,
        followers=[mounting_clip],
        cutters=[screw_hole_drill, alignment_slot],
        non_production_parts=[reference_screw],
        follower_names=["clip"],
        cutter_names=["screw_hole", "alignment"],
        non_production_names=["m3_screw"]
    )

    # Use bracket to cut holes in a panel
    prepared_panel = bracket.use_as_cutter_on(panel)

    # Access components by name
    clip_index = bracket.get_follower_index_by_name("clip")
    ```

    **Transformations:**

    All transformation methods (translate, rotate, mirror) operate on all components
    simultaneously, maintaining their relative positions. Both in-place and functional
    transformation interfaces are supported through CAD-backend agnostic operations.
    """

    def __init__(
        self,
        leader,
        followers=None,
        cutters=None,
        non_production_parts=None,
        additional_data=None,
        follower_names=None,
        cutter_names=None,
        non_production_names=None,
    ):
        """Initialize a composite part with leader and associated components.

        Args:
            leader: The primary part shape (CAD object with fuse/cut operations)
            followers: Single part or list of parts that move with the leader
            cutters: Single part or list of shapes for cutting other parts
            non_production_parts: Single part or list of reference/visualization parts
            additional_data: Dictionary of arbitrary metadata for this composite
            follower_names: List of names for followers (must match count)
            cutter_names: List of names for cutters (must match count)
            non_production_names: List of names for non-production parts (must match count)

        Raises:
            AssertionError: If name list lengths don't match corresponding part counts
            AssertionError: If additional_data is not a dictionary
        """

        self.leader = leader
        # Store raw parts directly for convenience during construction
        self.followers = _ensure_list(followers)
        self.cutters = _ensure_list(cutters)
        self.non_production_parts = _ensure_list(non_production_parts)
        self.additional_data = additional_data if additional_data is not None else {}
        assert isinstance(self.additional_data, dict)

        self.follower_indices_by_name = {}
        if follower_names is not None:
            assert len(follower_names) == len(self.followers)
            for idx, name in enumerate(follower_names):
                self.follower_indices_by_name[name] = idx

        self.cutter_indices_by_name = {}
        if cutter_names is not None:
            assert len(cutter_names) == len(self.cutters)
            for idx, name in enumerate(cutter_names):
                self.cutter_indices_by_name[name] = idx

        self.non_production_indices_by_name = {}
        if non_production_names is not None:
            assert len(non_production_names) == len(self.non_production_parts)
            for idx, name in enumerate(non_production_names):
                self.non_production_indices_by_name[name] = idx

    def use_as_cutter_on(self, part):
        """Apply all cutters from this composite to a target part.

        This method sequentially cuts the target part with each cutter in this
        composite, effectively "preparing" the target for assembly with this part.

        This operation is CAD-backend agnostic - it assumes the part and cutters
        support the .cut() method without caring about the specific CAD implementation.

        Args:
            part: Target CAD part to be cut

        Returns:
            Modified copy of the target part with all cuts applied

        Example:
            # Prepare a panel for mounting a bracket
            prepared_panel = bracket.use_as_cutter_on(panel)
        """

        retval = copy_part(part)
        for cutter in self.cutters:
            retval = retval.cut(cutter)

        return retval

    def get_leader_as_part(self):
        """Get the leader part for direct geometric operations.

        Returns:
            The leader part object
        """
        return self.leader

    def get_follower_index_by_name(self, name):
        """Get the index of a named follower.

        Args:
            name: Name of the follower to find

        Returns:
            Index of the follower, or None if name not found
        """
        return self.follower_indices_by_name.get(name, None)

    def get_follower_part_by_name(self, name):
        index = self.get_follower_index_by_name(name)
        if index is not None:
            return self.followers[index]
        raise KeyError(
            f"Follower with name '{name}' not found. Available names: {list(self.follower_indices_by_name.keys())}"
        )

    def get_cutter_index_by_name(self, name):
        """Get the index of a named cutter.

        Args:
            name: Name of the cutter to find

        Returns:
            Index of the cutter, or None if name not found
        """
        return self.cutter_indices_by_name.get(name, None)

    def get_non_production_index_by_name(self, name):
        """Get the index of a named non-production part.

        Args:
            name: Name of the non-production part to find

        Returns:
            Index of the non-production part, or None if name not found
        """
        return self.non_production_indices_by_name.get(name, None)

    def get_non_production_part_by_name(self, name):
        index = self.get_non_production_index_by_name(name)
        if index is not None:
            return self.non_production_parts[index]
        raise KeyError(
            f"Non-production part with name '{name}' not found. Available names: {list(self.non_production_indices_by_name.keys())}"
        )

    def add_named_non_production_part(self, part, name):
        """Add a non-production part with a specified name.

        Args:
            part: The non-production part to add
            name: The name to associate with the non-production part

        Raises:
            ValueError: If the name already exists
        """
        if not isinstance(name, str):
            raise TypeError("Non-production part name must be a string.")
        if name in self.non_production_indices_by_name:
            raise ValueError(f"Non-production part name '{name}' already exists.")
        self.non_production_parts.append(part)
        self.non_production_indices_by_name[name] = len(self.non_production_parts) - 1

    def add_named_follower(self, follower, name):
        """Add a follower part with a specified name.

        Args:
            follower: The follower part to add
            name: The name to associate with the follower

        Raises:
            ValueError: If the name already exists
        """
        if not isinstance(name, str):
            raise TypeError("Follower name must be a string.")
        if name in self.follower_indices_by_name:
            raise ValueError(f"Follower name '{name}' already exists.")

        self.followers.append(follower)
        self.follower_indices_by_name[name] = len(self.followers) - 1

    def get_non_production_parts_fused(self):
        """Get all non-production parts fused into a single shape.

        Returns:
            Fused non-production parts, or empty collector if none exist
        """
        collector = PartCollector()
        for part in self.non_production_parts:
            collector.fuse(_unwrap_named_part(part))
        return collector.part if collector.part is not None else collector

    def leaders_followers_fused(self):
        """Get the leader and all followers fused into a single shape.

        This creates a unified geometry representing the complete manufacturable
        assembly (excluding cutters and non-production parts).

        Returns:
            Fused leader and followers, or empty collector if leader has no geometry
        """
        collector = PartCollector()
        for part in [_unwrap_named_part(self.leader)] + [
            _unwrap_named_part(follower) for follower in self.followers
        ]:
            collector.fuse(part)
        return collector.part if collector.part is not None else collector

    def merge_except_leader(self, other):
        """
        Merge in followers, cutter, and non-production parts from another composite, but not the leader.
        Keep the names and additional data from the other composite.
        """

        retval = LeaderFollowersCuttersPart(self.leader)
        # first add the named parts from self
        follower_indices_with_name = set()
        for name, idx in self.follower_indices_by_name.items():
            retval.add_named_follower(self.followers[idx], name)
            follower_indices_with_name.add(idx)

        # now add the unnamed followers from self
        for idx, follower in enumerate(self.followers):
            if idx not in follower_indices_with_name:
                retval.followers.append(_clone_part(follower))

        cutter_indices_with_name = set()
        for name, idx in self.cutter_indices_by_name.items():
            retval.add_named_cutter(self.cutters[idx], name)
            cutter_indices_with_name.add(idx)
        # now add the unnamed cutters from self
        for idx, cutter in enumerate(self.cutters):
            if idx not in cutter_indices_with_name:
                retval.cutters.append(_clone_part(cutter))

        non_production_indices_with_name = set()
        for name, idx in self.non_production_indices_by_name.items():
            retval.add_named_non_production_part(self.non_production_parts[idx], name)
            non_production_indices_with_name.add(idx)
        # now add the unnamed non-production parts from self
        for idx, non_production_part in enumerate(self.non_production_parts):
            if idx not in non_production_indices_with_name:
                retval.non_production_parts.append(_clone_part(non_production_part))

        # now add the named parts from other
        other_follower_indices_with_name = set()
        for name, idx in other.follower_indices_by_name.items():
            retval.add_named_follower(other.followers[idx], name)
            other_follower_indices_with_name.add(idx)

        # now add the unnamed followers from other
        for idx, follower in enumerate(other.followers):
            if idx not in other_follower_indices_with_name:
                retval.followers.append(_clone_part(follower))

        other_cutter_indices_with_name = set()
        for name, idx in other.cutter_indices_by_name.items():
            retval.add_named_cutter(other.cutters[idx], name)
            other_cutter_indices_with_name.add(idx)
        # now add the unnamed cutters from other
        for idx, cutter in enumerate(other.cutters):
            if idx not in other_cutter_indices_with_name:
                retval.cutters.append(_clone_part(cutter))
        other_non_production_indices_with_name = set()
        for name, idx in other.non_production_indices_by_name.items():
            retval.add_named_non_production_part(other.non_production_parts[idx], name)
            other_non_production_indices_with_name.add(idx)
        # now add the unnamed non-production parts from other
        for idx, non_production_part in enumerate(other.non_production_parts):
            if idx not in other_non_production_indices_with_name:
                retval.non_production_parts.append(_clone_part(non_production_part))

        return retval

    def add_named_cutter(self, cutter, name):
        """Add a cutter part with a specified name.

        Args:
            cutter: The cutter part to add
            name: The name to associate with the cutter

        Raises:
            ValueError: If the name already exists
        """
        if not isinstance(name, str):
            raise TypeError("Cutter name must be a string.")
        if name in self.cutter_indices_by_name:
            raise ValueError(f"Cutter name '{name}' already exists.")
        self.cutters.append(cutter)
        self.cutter_indices_by_name[name] = len(self.cutters) - 1

    def rename_follower(self, old_name, new_name):
        """Rename a follower part.

        Args:
            old_name: Current name of the follower
            new_name: New name to assign to the follower

        Raises:
            KeyError: If old_name does not exist
            ValueError: If new_name already exists
        """
        if old_name not in self.follower_indices_by_name:
            raise KeyError(f"Follower name '{old_name}' does not exist.")
        if new_name in self.follower_indices_by_name:
            raise ValueError(f"Follower name '{new_name}' already exists.")

        index = self.follower_indices_by_name.pop(old_name)
        self.follower_indices_by_name[new_name] = index

    def rename_cutter(self, old_name, new_name):
        """Rename a cutter part.

        Args:
            old_name: Current name of the cutter
            new_name: New name to assign to the cutter

        Raises:
            KeyError: If old_name does not exist
            ValueError: If new_name already exists
        """
        if old_name not in self.cutter_indices_by_name:
            raise KeyError(f"Cutter name '{old_name}' does not exist.")
        if new_name in self.cutter_indices_by_name:
            raise ValueError(f"Cutter name '{new_name}' already exists.")

        index = self.cutter_indices_by_name.pop(old_name)
        self.cutter_indices_by_name[new_name] = index

    def rename_non_production_part(self, old_name, new_name):
        """Rename a non-production part.

        Args:
            old_name: Current name of the non-production part
            new_name: New name to assign to the non-production part

        Raises:
            KeyError: If old_name does not exist
            ValueError: If new_name already exists
        """
        if old_name not in self.non_production_indices_by_name:
            raise KeyError(f"Non-production part name '{old_name}' does not exist.")
        if new_name in self.non_production_indices_by_name:
            raise ValueError(f"Non-production part name '{new_name}' already exists.")

        index = self.non_production_indices_by_name.pop(old_name)
        self.non_production_indices_by_name[new_name] = index

    def copy(self):
        """Create a deep copy of this composite part.

        All parts, names, and additional data are copied. Changes to the copy
        will not affect the original.

        Returns:
            New LeaderFollowersCuttersPart with copied components
        """
        result = LeaderFollowersCuttersPart(
            _clone_part(self.leader),
            [_clone_part(follower) for follower in self.followers],
            [_clone_part(cutter) for cutter in self.cutters],
            [_clone_part(non_prod) for non_prod in self.non_production_parts],
            additional_data=copy.deepcopy(self.additional_data),
        )
        result.follower_indices_by_name = self.follower_indices_by_name.copy()
        result.cutter_indices_by_name = self.cutter_indices_by_name.copy()
        result.non_production_indices_by_name = (
            self.non_production_indices_by_name.copy()
        )
        return result

    def BoundingBox(self):
        """Get the bounding box of the leader part.

        CAD-backend agnostic: Uses get_bounding_box() from the adapter layer
        to handle different backends' bounding box implementations uniformly.

        Returns:
            SimpleNamespace with xmin, xmax, ymin, ymax, zmin, zmax attributes
        """
        leader_bb = get_bounding_box(_unwrap_named_part(self.leader))
        return SimpleNamespace(
            xmin=leader_bb[0][0],
            xmax=leader_bb[1][0],
            ymin=leader_bb[0][1],
            ymax=leader_bb[1][1],
            zmin=leader_bb[0][2],
            zmax=leader_bb[1][2],
        )

    def Vertices(self):
        """Get all vertices of the leader part.

        CAD-backend agnostic: Uses get_vertex_points() from the adapter layer
        to handle different backends' vertex extraction implementations.

        Returns:
            List of vertex points from the leader geometry
        """
        return get_vertex_points(_unwrap_named_part(self.leader))

    def Vertexes(self):
        """Get all vertices of the leader part (alternative spelling).

        Returns:
            List of vertex points from the leader geometry
        """
        return get_vertex_points(_unwrap_named_part(self.leader))

    @property
    def BoundBox(self):
        """Get the bounding box of the leader part (CadQuery-compatible property).

        CAD-backend agnostic: Uses get_bounding_box() from the adapter layer
        while providing a CadQuery-style property interface. The implementation
        doesn't care about the specific backend - it just transforms the generic
        bounding box format to the expected property format.

        Returns:
            SimpleNamespace with XMin, YMin, ZMin, XMax, YMax, ZMax attributes
        """
        leader_bb = get_bounding_box(_unwrap_named_part(self.leader))

        return SimpleNamespace(
            XMin=leader_bb[0][0],
            YMin=leader_bb[0][1],
            ZMin=leader_bb[0][2],
            XMax=leader_bb[1][0],
            YMax=leader_bb[1][1],
            ZMax=leader_bb[1][2],
        )

    def fuse(
        self,
        other,
    ):
        """Fuse this composite with another part or composite.

        When fusing with another LeaderFollowersCuttersPart:
        - Leaders are geometrically fused
        - All followers, cutters, and non-production parts are merged
        - Named components from both composites are combined
        - Name collisions raise ValueError

        When fusing with a simple part:
        - Leader is fused with the part
        - All associated components are preserved with their names

        This operation is CAD-backend agnostic - it delegates to the underlying
        CAD objects' .fuse() methods without backend-specific logic.

        Args:
            other: Another LeaderFollowersCuttersPart or simple CAD part

        Returns:
            New composite with fused geometry and merged components

        Raises:
            ValueError: If named components have conflicting names
        """

        if isinstance(other, LeaderFollowersCuttersPart):
            new_leader = _unwrap_named_part(self.leader).fuse(
                _unwrap_named_part(other.leader)
            )
            new_followers = [_clone_part(f) for f in (self.followers + other.followers)]

            # Merge follower names, checking for collisions
            new_follower_indices_by_name = self.follower_indices_by_name.copy()
            for other_name, other_idx in other.follower_indices_by_name.items():
                if other_name in new_follower_indices_by_name:
                    raise ValueError(
                        f"Follower name collision: '{other_name}' already exists"
                    )
                new_follower_indices_by_name[other_name] = other_idx + len(
                    self.followers
                )

            new_cutters = [_clone_part(c) for c in (self.cutters + other.cutters)]

            # Merge cutter names, checking for collisions
            new_cutter_indices_by_name = self.cutter_indices_by_name.copy()
            for other_name, other_idx in other.cutter_indices_by_name.items():
                if other_name in new_cutter_indices_by_name:
                    raise ValueError(
                        f"Cutter name collision: '{other_name}' already exists"
                    )
                new_cutter_indices_by_name[other_name] = other_idx + len(self.cutters)

            new_non_prod = [
                _clone_part(n)
                for n in (self.non_production_parts + other.non_production_parts)
            ]

            # Merge non-production part names, checking for collisions
            new_non_production_indices_by_name = (
                self.non_production_indices_by_name.copy()
            )
            for other_name, other_idx in other.non_production_indices_by_name.items():
                if other_name in new_non_production_indices_by_name:
                    raise ValueError(
                        f"Non-production part name collision: '{other_name}' already exists"
                    )
                new_non_production_indices_by_name[other_name] = other_idx + len(
                    self.non_production_parts
                )

            result = LeaderFollowersCuttersPart(
                new_leader, new_followers, new_cutters, new_non_prod
            )
            result.follower_indices_by_name = new_follower_indices_by_name
            result.cutter_indices_by_name = new_cutter_indices_by_name
            result.non_production_indices_by_name = new_non_production_indices_by_name
            return result

        other_shape = _unwrap_named_part(other)
        new_leader = _unwrap_named_part(self.leader).fuse(other_shape)
        new_additinoal_data = copy.deepcopy(self.additional_data)
        new_additinoal_data.update(
            other.additional_data if hasattr(other, "additional_data") else {}
        )

        result = LeaderFollowersCuttersPart(
            new_leader,
            [_clone_part(f) for f in self.followers],
            [_clone_part(c) for c in self.cutters],
            [_clone_part(n) for n in self.non_production_parts],
            additional_data=new_additinoal_data,
        )
        result.follower_indices_by_name = self.follower_indices_by_name.copy()
        result.cutter_indices_by_name = self.cutter_indices_by_name.copy()
        result.non_production_indices_by_name = (
            self.non_production_indices_by_name.copy()
        )
        return result

    def cut(self, other):
        """Cut this composite with another part, removing material from the leader.

        The cutting operation only affects the leader geometry. All followers,
        cutters, and non-production parts from the original composite are preserved.
        The cutting part does NOT contribute its components to the result.

        This operation is CAD-backend agnostic - it delegates to the underlying
        CAD objects' .cut() methods without backend-specific conditional logic.

        Args:
            other: Another LeaderFollowersCuttersPart or simple CAD part to cut with

        Returns:
            New composite with cut leader and preserved original components

        Raises:
            TypeError: If other part doesn't support cut operations

        Example:
            # Remove material from bracket using a drill shape
            drilled_bracket = bracket.cut(drill_cylinder)
        """

        leader_shape = _unwrap_named_part(self.leader)

        if isinstance(other, LeaderFollowersCuttersPart):
            other_leader = _unwrap_named_part(other.leader)
            new_leader = leader_shape.cut(other_leader)

            # Keep only the original part's followers, cutters, and non-production parts
            # The cutting part doesn't contribute its components to the result
            result = LeaderFollowersCuttersPart(
                new_leader,
                [_clone_part(follower) for follower in self.followers],
                [_clone_part(cutter) for cutter in self.cutters],
                [_clone_part(non_prod) for non_prod in self.non_production_parts],
                additional_data=copy.deepcopy(self.additional_data),
            )
            result.follower_indices_by_name = self.follower_indices_by_name.copy()
            result.cutter_indices_by_name = self.cutter_indices_by_name.copy()
            result.non_production_indices_by_name = (
                self.non_production_indices_by_name.copy()
            )
            return result

        other_shape = _unwrap_named_part(other)
        try:
            new_leader = leader_shape.cut(other_shape)
        except AttributeError as exc:
            raise TypeError(
                "other must be a LeaderFollowersCuttersPart or provide a cut() operation"
            ) from exc

        result = LeaderFollowersCuttersPart(
            new_leader,
            [_clone_part(follower) for follower in self.followers],
            [_clone_part(cutter) for cutter in self.cutters],
            [_clone_part(non_prod) for non_prod in self.non_production_parts],
            additional_data=copy.deepcopy(self.additional_data),
        )
        result.follower_indices_by_name = self.follower_indices_by_name.copy()
        result.cutter_indices_by_name = self.cutter_indices_by_name.copy()
        result.non_production_indices_by_name = (
            self.non_production_indices_by_name.copy()
        )
        return result

    def translate(self, *args):
        """Translate all parts in this composite by the same vector.

        This is an in-place operation that moves the leader and all associated
        parts together, maintaining their relative positions.

        CAD-backend agnostic: Uses translate_part_native() for the leader and
        delegates to each part's .translate() method. The argument format (*args)
        is intentionally flexible to accommodate different backends' parameter
        conventions without requiring backend-specific conditional logic.

        Args:
            *args: Translation vector - format depends on CAD backend
                   (e.g., (x,y,z) tuple or separate x,y,z arguments)

        Returns:
            Self (for method chaining)
        """
        self.leader = translate_part_native(self.leader, *args)
        self.followers = [follower.translate(*args) for follower in self.followers]
        self.cutters = [cutter.translate(*args) for cutter in self.cutters]
        self.non_production_parts = [
            part.translate(*args) for part in self.non_production_parts
        ]
        return self

    def rotate(self, *args):
        """Rotate all parts in this composite around the same axis.

        This is an in-place operation that rotates the leader and all associated
        parts together, maintaining their relative positions.

        CAD-backend agnostic: Uses rotate_part_native() for the leader and
        delegates to each part's .rotate() method. The argument format (*args)
        is intentionally flexible to accommodate different backends' parameter
        conventions (e.g., CadQuery vs FreeCAD rotation signatures) without
        requiring backend-specific conditional logic.

        Args:
            *args: Rotation parameters - format depends on CAD backend
                   (e.g., center_point, axis_vector, angle for some backends)

        Returns:
            Self (for method chaining)
        """
        self.leader = rotate_part_native(self.leader, *args)
        self.followers = [follower.rotate(*args) for follower in self.followers]
        self.cutters = [cutter.rotate(*args) for cutter in self.cutters]
        self.non_production_parts = [
            part.rotate(*args) for part in self.non_production_parts
        ]
        return self

    def scale(self, factor, center=(0.0, 0.0, 0.0)):
        """Scale all parts in this composite around the same center.

        This is an in-place operation that scales the leader and all associated
        parts together, maintaining their relative positions.

        CAD-backend agnostic: Uses scale_part_native() for the leader and
        applies the same scaling to all subparts, preserving NamedPart wrappers.

        Args:
            factor: Uniform scale factor or (x, y, z) tuple
            center: Center point for scaling as (x, y, z) tuple

        Returns:
            Self (for method chaining)
        """

        def scale_component(component):
            if isinstance(component, NamedPart):
                return component.scale(factor, center=center)
            return scale_part_native(component, factor, center=center)

        self.leader = scale_part_native(self.leader, factor, center=center)
        self.followers = [scale_component(follower) for follower in self.followers]
        self.cutters = [scale_component(cutter) for cutter in self.cutters]
        self.non_production_parts = [
            scale_component(part) for part in self.non_production_parts
        ]
        return self

    def mirror(self, *args, **kwargs):
        """Mirror all parts in this composite across a plane.

        This is an in-place operation that mirrors the leader and all associated
        parts together, maintaining their relative positions.

        CAD-backend agnostic: Uses mirror_part_native() for the leader and
        delegates to each part's .mirror() method. The flexible signature
        (*args, **kwargs) accommodates different backends' parameter conventions
        without requiring backend-specific conditional logic.

        Args:
            *args: Mirror plane parameters - format depends on CAD backend
            **kwargs: Additional mirror options - varies by backend

        Returns:
            Self (for method chaining)
        """
        self.leader = mirror_part_native(self.leader, *args, **kwargs)
        self.followers = [
            follower.mirror(*args, **kwargs) for follower in self.followers
        ]
        self.cutters = [cutter.mirror(*args, **kwargs) for cutter in self.cutters]
        self.non_production_parts = [
            part.mirror(*args, **kwargs) for part in self.non_production_parts
        ]
        return self

    def reconstruct(self, transformed_result=None):
        """Reconstruct this composite after transformation operations.

        This is a key method for maintaining CAD-backend agnosticism. When functional
        transformations are applied (which return new geometry objects), this method
        rebuilds the composite structure while preserving names and metadata.

        This pattern allows the class to work with any CAD backend that supports
        the transformation operations, without requiring knowledge of the specific
        backend's object lifecycle or transformation return patterns.

        Args:
            transformed_result: Optional pre-transformed composite to use as base

        Returns:
            New composite with preserved names and metadata
        """

        if transformed_result is not None:
            # Use the transformation result if provided
            result = LeaderFollowersCuttersPart(
                transformed_result.leader,
                [follower for follower in transformed_result.followers],
                [cutter for cutter in transformed_result.cutters],
                [part for part in transformed_result.non_production_parts],
                additional_data=copy.deepcopy(self.additional_data),
            )
            result.follower_indices_by_name = self.follower_indices_by_name.copy()
            result.cutter_indices_by_name = self.cutter_indices_by_name.copy()
            result.non_production_indices_by_name = (
                self.non_production_indices_by_name.copy()
            )
            return result

        else:
            result = LeaderFollowersCuttersPart(
                _clone_part(self.leader),
                [_clone_part(follower) for follower in self.followers],
                [_clone_part(cutter) for cutter in self.cutters],
                [_clone_part(part) for part in self.non_production_parts],
                additional_data=copy.deepcopy(self.additional_data),
            )
            result.follower_indices_by_name = self.follower_indices_by_name.copy()
            result.cutter_indices_by_name = self.cutter_indices_by_name.copy()
            result.non_production_indices_by_name = (
                self.non_production_indices_by_name.copy()
            )
            return result


def _unwrap_named_part(part):
    """Extract the underlying CAD part from a NamedPart wrapper.

    This helper function maintains CAD-backend agnosticism by providing a
    uniform interface to access the actual geometry object, regardless of
    whether it's wrapped in a NamedPart or is a direct CAD backend object.

    Args:
        part: Either a NamedPart or direct CAD part

    Returns:
        The underlying CAD part object
    """
    if isinstance(part, NamedPart):
        return part.part
    return part


def _clone_part(part):
    """Create a deep copy of a part, handling both NamedPart and direct CAD parts.

    This helper function maintains CAD-backend agnosticism by using polymorphic
    copy operations - NamedPart.copy() for wrapped parts, or copy_part() from
    the adapter layer for direct CAD objects. No backend-specific logic needed.

    Args:
        part: Either a NamedPart or direct CAD part

    Returns:
        A copied version of the part
    """
    if isinstance(part, NamedPart):
        return part.copy()
    return copy_part(part)


def reset_to_original_orientation(leader_followers_cutters_part):
    """Reset a composite part to its original orientation by creating a copy.

    This is a convenience function that creates a fresh copy of the composite,
    effectively "undoing" any transformations applied to the original.

    Args:
        leader_followers_cutters_part: The composite part to reset

    Returns:
        A copy of the composite part in its original state
    """

    return leader_followers_cutters_part.copy()
