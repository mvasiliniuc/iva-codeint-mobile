from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DatasetTokenizationArguments:
    """
    Configuration for data pretokenization.
    """
    tokenizer_dir: Optional[str] = field(
        default="mvasiliniuc/iva-codeint-swift-small", metadata={"help": "Name or path to the tokenizer."}
    )
    dataset_name: Optional[str] = field(
        default="mvasiliniuc/iva-swift-codeint-clean-train", metadata={"help": "Name or path to the dataset to pretokenize."}
    )
    tokenized_data_repo: Optional[str] = field(
        default="mvasiliniuc/iva-swift-codeint-clean-train-tokenized", metadata={"help": "Repo name of the pretokenized data."}
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})
