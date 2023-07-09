from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvaluationArguments:
    """
    Configuration for evaluating model.
    """
    model_ckpt: Optional[str] = field(
        default="mvasiliniuc/iva-codeint-swift-small", metadata={"help": "Model name or path of model to be evaluated."}
    )
    dataset_name: Optional[str] = field(
        default="mvasiliniuc/iva-swift-codeint-clean", metadata={"help": "Name or path of validation dataset."}
    )
    batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size used for evaluation."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Length of sequences to be evaluated."})
    seed: Optional[int] = field(default=1, metadata={"help": "Random seed used for evaluation."})