import logging
import torch
from accelerate import Accelerator
from model_validation_args import EvaluationArguments
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from constantLengthDataset import ConstantLengthDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

def create_dataloader(args):
    ds_kwargs = {"streaming": True}
    valid_data = load_dataset(args.dataset_name, split="train", **ds_kwargs)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data, seq_length=args.seq_length)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    return eval_dataloader

def evaluate(args):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

# Setup Accelerator
accelerator = Accelerator()
# Parse configuration
parser = HfArgumentParser(EvaluationArguments)
args = parser.parse_args()
set_seed(args.seed)
# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
# Load dataset and dataloader
eval_dataloader = create_dataloader(args)
# Prepare everything with our `accelerator`.
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
# Evaluate and save the last checkpoint
logger.info("Evaluating and saving model after training")
eval_loss, perplexity = evaluate(args)
logger.info(f"loss/eval: {eval_loss}, perplexity: {perplexity}")
