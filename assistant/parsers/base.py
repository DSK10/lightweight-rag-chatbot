from dataclasses import dataclass, field


@dataclass
class Section:
    title: str
    content: str
    metadata: dict = field(default_factory=dict)

    