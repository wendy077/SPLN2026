import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from retriever_hybrid import HybridRetriever, load_chunks, load_model_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "sbert_chunk_embeddings.npy"

BASE_QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
FINETUNED_QA_MODEL_DIR = PROJECT_ROOT / "models" / "qa_finetuned"


def get_qa_model_name() -> str:
    if (FINETUNED_QA_MODEL_DIR / "config.json").exists():
        return str(FINETUNED_QA_MODEL_DIR)

    return BASE_QA_MODEL_NAME


def build_context(result: dict) -> str:
    metadata = [
        f"The document title is {result['title']}.",
        f"The document type is {result['type']}.",
        f"The source is {result['source']}.",
    ]

    if result.get("artist"):
        metadata.append(f"The artist associated with this document is {result['artist']}.")

    return " ".join(metadata) + "\n\n" + result["text"]

def postprocess_answer(answer: str) -> str:
    answer = answer.strip()

    # Remove artefactos causados por metadados.
    answer = answer.replace(" Type", "")
    answer = answer.replace(" Source", "")
    answer = answer.replace(" Document", "")

    answer = answer.strip(" .,:;!?\"'()[]{}")
    return answer


def metadata_answer_if_possible(question: str, chunk: dict) -> dict | None:
    q = question.lower()

    # Perguntas sobre quem lançou um álbum.
    if "who released" in q and chunk.get("type") == "album" and chunk.get("artist"):
        return {
            "answer": chunk["artist"],
            "qa_score": 1.0,
            "answer_method": "metadata",
        }

    # Perguntas sobre que álbum contém uma música.
    if ("which album" in q or "what album" in q) and chunk.get("type") == "album":
        return {
            "answer": chunk["title"],
            "qa_score": 1.0,
            "answer_method": "metadata",
        }

    # Perguntas sobre género.
    if "what genre" in q and chunk.get("type") == "genre":
        return {
            "answer": chunk["title"],
            "qa_score": 1.0,
            "answer_method": "metadata",
        }

    return None

class ExtractiveQAModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    def answer(self, question: str, context: str, max_answer_tokens: int = 30) -> dict:
        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        input_ids = encoded["input_ids"][0]

        # Queremos respostas apenas no contexto, não na pergunta nem em tokens especiais.
        sequence_ids = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=512,
        ).sequence_ids()

        valid_context_mask = torch.tensor(
            [sid == 1 for sid in sequence_ids],
            device=self.device,
            dtype=torch.bool,
        )

        start_logits = start_logits.masked_fill(~valid_context_mask, -1e9)
        end_logits = end_logits.masked_fill(~valid_context_mask, -1e9)

        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)

        top_starts = torch.topk(start_probs, k=min(10, len(start_probs))).indices
        top_ends = torch.topk(end_probs, k=min(10, len(end_probs))).indices

        best_score = -1.0
        best_start = None
        best_end = None

        for start in top_starts:
            for end in top_ends:
                start_i = int(start.item())
                end_i = int(end.item())

                if end_i < start_i:
                    continue

                if end_i - start_i + 1 > max_answer_tokens:
                    continue

                score = float(start_probs[start_i] * end_probs[end_i])

                if score > best_score:
                    best_score = score
                    best_start = start_i
                    best_end = end_i

        if best_start is None or best_end is None:
            return {
                "answer": "",
                "score": 0.0,
            }

        answer_ids = input_ids[best_start:best_end + 1]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "score": best_score,
        }


def answer_question(
    question: str,
    retriever: HybridRetriever,
    qa_model: ExtractiveQAModel,
    top_k_retrieval: int = 8,
    top_k_answers: int = 5,
    alpha: float = 0.6,
    filter_type: str | None = None,
    unique_docs: bool = False,
) -> list[dict]:
    retrieved_chunks = retriever.search(
        query=question,
        top_k=top_k_retrieval,
        alpha=alpha,
        unique_docs=unique_docs,
        filter_type=filter_type,
    )

    candidate_answers = []

    for chunk in retrieved_chunks:
        context = build_context(chunk)

        metadata_result = metadata_answer_if_possible(question, chunk)

        if metadata_result is not None:
            qa_result = metadata_result
        else:
            try:
                qa_result = qa_model.answer(
                    question=question,
                    context=context,
                )
                qa_result["answer_method"] = "model"
            except Exception as exc:
                print(f"[WARN] QA failed for chunk {chunk['chunk_id']}: {exc}")
                continue

        answer = postprocess_answer(qa_result.get("answer", ""))
        qa_score = float(qa_result.get("qa_score", qa_result.get("score", 0.0)))
        retrieval_score = float(chunk["score"])

        if not answer:
            continue

        final_score = 0.7 * qa_score + 0.3 * retrieval_score

        candidate_answers.append({
            "answer": answer,
            "final_score": final_score,
            "qa_score": qa_score,
            "retrieval_score": retrieval_score,
            "answer_method": qa_result.get("answer_method", "model"),
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "type": chunk["type"],
            "source": chunk["source"],
            "url": chunk["url"],
            "context": chunk["text"],
        })

    candidate_answers.sort(key=lambda x: x["final_score"], reverse=True)

    unique_answers = []
    seen = set()

    for item in candidate_answers:
        normalized_answer = item["answer"].lower().strip(" .,:;!?\"'()[]{}")

        if normalized_answer in seen:
            continue

        seen.add(normalized_answer)
        unique_answers.append(item)

        if len(unique_answers) >= top_k_answers:
            break

    return unique_answers


def print_answers(question: str, answers: list[dict]) -> None:
    print("=" * 100)
    print(f"Question: {question}")

    if not answers:
        print("No answer found.")
        return

    best = answers[0]

    print()
    print("Best answer:")
    print(best["answer"])
    print()
    print(f"Final score: {best['final_score']:.4f}")
    print(f"QA score: {best['qa_score']:.4f}")
    print(f"Retrieval score: {best['retrieval_score']:.4f}")
    print(f"Answer method: {best.get('answer_method', 'model')}")
    print(f"Source title: {best['title']}")
    print(f"Source type: {best['type']}")
    print(f"Source: {best['source']}")
    print(f"URL: {best['url']}")

    print()
    print("Supporting context:")
    print(best["context"][:1000])

    if len(answers) > 1:
        print()
        print("=" * 100)
        print("Other candidate answers:")

        for idx, item in enumerate(answers[1:], start=2):
            print("-" * 100)
            print(f"{idx}. {item['answer']}")
            print(f"Final score: {item['final_score']:.4f}")
            print(f"QA score: {item['qa_score']:.4f}")
            print(f"Retrieval score: {item['retrieval_score']:.4f}")
            print(f"Answer method: {item.get('answer_method', 'model')}")
            print(f"Source title: {item['title']}")
            print(f"URL: {item['url']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)

    parser.add_argument("--top-k-retrieval", type=int, default=8)
    parser.add_argument("--top-k-answers", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--unique-docs", action="store_true")

    parser.add_argument(
        "--filter-type",
        type=str,
        default=None,
        choices=["artist", "album", "genre", "festival", "award", "event", "movement"],
    )

    args = parser.parse_args()

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    model_name = load_model_name()

    print(f"Loading retriever model: {model_name}")
    retriever = HybridRetriever(chunks, embeddings, model_name)

    qa_model_name = get_qa_model_name()
    print(f"Loading QA model: {qa_model_name}")
    qa_model = ExtractiveQAModel(qa_model_name)

    answers = answer_question(
        question=args.question,
        retriever=retriever,
        qa_model=qa_model,
        top_k_retrieval=args.top_k_retrieval,
        top_k_answers=args.top_k_answers,
        alpha=args.alpha,
        filter_type=args.filter_type,
        unique_docs=args.unique_docs,
    )

    print_answers(args.question, answers)


if __name__ == "__main__":
    main()