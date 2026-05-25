import argparse
import json
import random
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_MODEL = "distilbert-base-cased"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "qa_finetuned"


# ---------------------------------------------------------------------------
# EM / F1 helpers (seguem a implementação oficial do SQuAD 2.0)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def compute_exact(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def predict_answer(model, tokenizer, question: str, context: str, device: torch.device, max_length: int = 384) -> str:
    """Extrai um span de resposta do modelo QA."""
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    start_idx = int(outputs.start_logits[0].argmax())
    end_idx = int(outputs.end_logits[0].argmax())

    if end_idx < start_idx:
        end_idx = start_idx

    answer_ids = encoded["input_ids"][0][start_idx: end_idx + 1]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

def evaluate_squad_em_f1(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    max_samples: int | None = None,
    max_length: int = 384,
) -> dict:
    """Calcula EM e F1 no split de validação do SQuAD."""
    model.eval()
    exact_scores = []
    f1_scores = []

    if max_samples is None or max_samples <= 0:
        samples = dataset
    else:
        samples = dataset.select(range(min(max_samples, len(dataset))))

    for example in tqdm(samples, desc="Evaluating EM/F1"):
        question = example["question"]
        context = example["context"]
        answers = example["answers"]["text"]

        if not answers:
            continue

        prediction = predict_answer(model, tokenizer, question, context, device, max_length)

        em = max(compute_exact(prediction, ref) for ref in answers)
        f1 = max(compute_f1(prediction, ref) for ref in answers)

        exact_scores.append(em)
        f1_scores.append(f1)

    return {
        "exact_match": round(100.0 * sum(exact_scores) / len(exact_scores), 2) if exact_scores else 0.0,
        "f1": round(100.0 * sum(f1_scores) / len(f1_scores), 2) if f1_scores else 0.0,
        "num_samples": len(exact_scores),
    }

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def prepare_train_features(examples, tokenizer, max_length: int, doc_stride: int):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]
        end_char = start_char + len(answer_text)

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Se a resposta não estiver totalmente dentro deste span, usa CLS.
        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples


def make_dataloader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]

    if "token_type_ids" in dataset.column_names:
        columns.append("token_type_ids")

    dataset.set_format(type="torch", columns=columns)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_loss(model, dataloader, device: torch.device) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.item()))

    model.train()

    if not losses:
        return 0.0

    return float(np.mean(losses))


def train(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading dataset: squad")
    dataset = load_dataset("squad")

    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))

    if args.max_eval_samples:
        dataset["validation"] = dataset["validation"].select(range(min(args.max_eval_samples, len(dataset["validation"]))))

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Eval samples: {len(dataset['validation'])}")

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    print(f"Loading model: {args.base_model}")
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)
    model.to(device)
    model.train()

    print("Tokenizing train split")
    train_features = dataset["train"].map(
        lambda examples: prepare_train_features(
            examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print("Tokenizing validation split")
    eval_features = dataset["validation"].map(
        lambda examples: prepare_train_features(
            examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
        ),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    train_loader = make_dataloader(train_features, batch_size=args.batch_size, shuffle=True)
    eval_loader = make_dataloader(eval_features, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    global_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        progress = tqdm(train_loader, desc="Training")

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "step": global_step,
            })

            if args.max_steps and global_step >= args.max_steps:
                break

        eval_loss = evaluate_loss(model, eval_loader, device)
        print(f"Eval loss after epoch {epoch + 1}: {eval_loss:.4f}")

        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"\nSaving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # -------------------------------------------------------------------
    # Avaliação EM/F1 — modelo fine-tuned vs. modelo base
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Evaluating fine-tuned model on SQuAD validation (EM / F1)...")
    print("=" * 70)

    finetuned_metrics = evaluate_squad_em_f1(
        model, tokenizer, dataset["validation"], device,
        max_samples=args.max_eval_samples,
        max_length=args.max_length,
    )
    print(f"Fine-tuned  —  EM: {finetuned_metrics['exact_match']:.2f}  |  F1: {finetuned_metrics['f1']:.2f}  (n={finetuned_metrics['num_samples']})")

    print(f"\nLoading baseline model for comparison: distilbert-base-cased-distilled-squad")
    baseline_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    baseline_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    baseline_model.to(device)
    baseline_metrics = evaluate_squad_em_f1(
        baseline_model, baseline_tokenizer, dataset["validation"], device,
        max_samples=args.max_eval_samples,
        max_length=args.max_length,
    )
    print(f"Baseline    —  EM: {baseline_metrics['exact_match']:.2f}  |  F1: {baseline_metrics['f1']:.2f}  (n={baseline_metrics['num_samples']})")

    delta_em = finetuned_metrics["exact_match"] - baseline_metrics["exact_match"]
    delta_f1 = finetuned_metrics["f1"] - baseline_metrics["f1"]
    print(f"\nDelta (fine-tuned − baseline)  —  EM: {delta_em:+.2f}  |  F1: {delta_f1:+.2f}")
    print("=" * 70)

    metadata = {
        "base_model": args.base_model,
        "dataset": "squad",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "doc_stride": args.doc_stride,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "eval_metrics": {
            "finetuned": finetuned_metrics,
            "baseline": baseline_metrics,
            "delta_em": round(delta_em, 2),
            "delta_f1": round(delta_f1, 2),
        },
    }

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Número máximo de exemplos de treino. Usa o dataset completo se não for definido.",
    )

    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Número máximo de exemplos de validação. Usa o dataset completo se não for definido.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())