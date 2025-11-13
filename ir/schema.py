from dataclasses import dataclass
from typing import Any, List, Dict


@dataclass
class OperatorIR:
    name: str
    inputs: List[Any]
    outputs: List[Any]
    properties: Dict[str, Any]


@dataclass
class ModelIR:
    name: str
    layers: list
    connections: list


@dataclass
class ApplicationIR:
    name: str
    purpose: str
    input_format: str
    output_format: str
