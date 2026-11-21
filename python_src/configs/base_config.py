r"""
Base Configuration DataClass
============================
Provides unified dictionary serialization, deserialization, and overriding capabilities.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Type, TypeVar

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base dataclass for all TrajectoryLNAS configurations."""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration instance to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """Create a configuration instance from a dictionary, ignoring extra keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def copy_with_overrides(self: T, **kwargs: Any) -> T:
        """Return a copy of the configuration with specified overrides applied."""
        current = self.to_dict()
        current.update(kwargs)
        return self.from_dict(current)
