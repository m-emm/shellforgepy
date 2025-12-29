import hashlib

DEFAULT_RESOLUTION = 1_000_000


class PartParameters:

    def __init__(self, parameters: dict):

        # validate that parameters are all of type int, float, str, or bool
        for key, value in parameters.items():
            if not isinstance(value, (int, float, str, bool)):
                raise ValueError(
                    f"Parameter '{key}' has invalid type {type(value)}. "
                    "Only int, float, str, and bool are allowed."
                )

        self.parameters = parameters

    def __getattr__(self, item):
        if item in self.parameters:
            return self.parameters[item]
        else:
            raise AttributeError(f"'PartParameters' object has no attribute '{item}'")

    def parameters_hash(self) -> str:
        """Generate a hash representing the current set of parameters."""

        # Convert all floats to integers by multiplying by the resoution, given in the parameters, or the default

        resolution = self.parameters.get("_hash_resolution", DEFAULT_RESOLUTION)

        hash_input_parts = []
        for key in sorted(self.parameters.keys()):
            value = self.parameters[key]
            if isinstance(value, float):
                int_value = int(round(value * resolution))
                hash_input_parts.append(f"{key}={int_value}")
            else:
                hash_input_parts.append(f"{key}={value}")
        hash_input = "|".join(hash_input_parts)
        hash_bytes = hash_input.encode("utf-8")
        hash_digest = hashlib.sha256(hash_bytes).hexdigest()
        return hash_digest

    def modified_copy(self, modifications: dict) -> "PartParameters":
        """Create a modified copy of the current PartParameters with given modifications."""

        new_parameters = self.parameters.copy()
        new_parameters.update(modifications)
        return PartParameters(new_parameters)
