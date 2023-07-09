from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelInitializationArguments:
    """
    Configuration for initializing new model.
    """
    config_name: Optional[str] = field(
        default="gpt2", metadata={"help": "Configuration to use for model initialization."}
    )
    tokenizer_name: Optional[str] = field(
        default="mvasiliniuc/iva-codeint-kotlin-small", metadata={"help": "Tokenizer attached to model."}
    )
    model_name: Optional[str] = field(default="iva-codeint-kotlin-small", metadata={"help": "Name of the created model."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})