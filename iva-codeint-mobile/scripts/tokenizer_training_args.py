from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training.
    """
    base_tokenizer: Optional[str] = field(
        default="gpt2", metadata={"help": "Base tokenizer to build new tokenizer from."}
    )
    dataset_name: Optional[str] = field(
        default="mvasiliniuc/iva-kotlin-codeint-clean", metadata={"help": "Dataset to train tokenizer on."}
    )
    text_column: Optional[str] = field(default="content", metadata={"help": "Column containing text data to process."})
    vocab_size: Optional[int] = field(default=200_000, metadata={"help": "Number of examples to train tokenizer on."})
    n_examples: Optional[int] = field(
        default=32768, metadata={"help": "Number of examples to train the tokenizer on."}
    )
    tokenizer_name: Optional[str] = field(default="iva-codeint-kotlin-small", metadata={"help": "Name of new tokenizer."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})