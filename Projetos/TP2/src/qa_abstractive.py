import argparse
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from retriever_hybrid import HybridRetriever, load_chunks, load_model_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "sbert_chunk_embeddings.npy"

GEN_MODEL_NAME = "google/flan-t5-base"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def clean_generated_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.strip('"')
    answer = answer.strip("'")
    answer = answer.strip()
    return answer


def metadata_answer_if_possible(question: str, retrieved_chunks: list[dict]) -> str | None:
    q = question.lower()

    if not retrieved_chunks:
        return None

    best = retrieved_chunks[0]

    if "who released" in q and best.get("type") == "album" and best.get("artist"):
        return best["artist"]

    if ("which album" in q or "what album" in q) and best.get("type") == "album":
        return best["title"]

    if "what genre" in q and best.get("type") == "genre":
        return best["title"]

    return None

def split_sentences(text: str) -> list[str]:
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def is_explanation_question(question: str) -> bool:
    q = question.lower()
    return (
        q.startswith("why")
        or q.startswith("how")
        or q.startswith("explain")
        or "important" in q
        or "significant" in q
        or "influence" in q
        or "impact" in q
    )


def select_evidence_sentences(question: str, retrieved_chunks: list[dict], max_sentences: int = 8) -> list[str]:
    question_terms = set(
        word.lower().strip(".,:;!?()[]{}\"'")
        for word in question.split()
        if len(word) > 3
    )

    importance_terms = {
        "important",
        "importance",
        "significant",
        "legacy",
        "impact",
        "critical",
        "critics",
        "ranked",
        "awards",
        "award",
        "grammy",
        "racial",
        "empowerment",
        "conscious",
        "experimental",
        "artistic",
        "generation",
        "revered",
        "respected",
        "acclaim",
    }

    noisy_terms = {
        "first week",
        "spotify",
        "billboard 200",
        "sales",
        "sold",
        "united kingdom",
        "australia",
        "united states",
        "david bowie",
    }

    candidates = []

    for chunk in retrieved_chunks:
        for sentence in split_sentences(chunk["text"]):
            sentence = sentence.strip()
            sentence_lower = sentence.lower()

            # Evita frases demasiado curtas ou pouco informativas.
            if len(sentence.split()) < 8:
                continue

            # Evita frases cortadas/incompletas.
            if not sentence.endswith((".", "!", "?", '"')):
                continue

            # Evita evidência irrelevante para perguntas explicativas.
            if any(term in sentence_lower for term in noisy_terms):
                continue

            score = 0

            for term in question_terms:
                if term in sentence_lower:
                    score += 2

            for term in importance_terms:
                if term in sentence_lower:
                    score += 3

            if chunk["title"].lower() in sentence_lower:
                score += 1

            if score > 0:
                candidates.append((score, sentence, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()

    for _, sentence, chunk in candidates:
        normalized = sentence.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        selected.append(
            f"Title: {chunk['title']}. Evidence: {sentence}"
        )

        if len(selected) >= max_sentences:
            break

    return selected

def is_bad_explanation_answer(answer: str, question: str) -> bool:
    answer_lower = answer.lower().strip()
    question_lower = question.lower().strip()

    if not answer_lower:
        return True

    if len(answer_lower.split()) < 8:
        return True

    if answer_lower.endswith("2016") or answer_lower.endswith("2015"):
        return True

    if answer_lower in question_lower:
        return True

    bad_fragments = [
        "united kingdom",
        "australia",
        "united states",
        "was an influence on david bowie",
        "is considered important",
    ]

    return any(fragment in answer_lower for fragment in bad_fragments)


def synthesize_explanation_from_evidence(question: str, evidence_sentences: list[str]) -> str:
    """Build a generic explanation answer by extracting key points from evidence sentences.

    This function is a domain-agnostic fallback: it identifies informative sentences from
    the retrieved evidence and assembles them into a concise answer. It does not assume
    any specific artist, album or topic.
    """
    if not evidence_sentences:
        return "The retrieved documents do not contain enough information to answer this question."

    evidence_text = " ".join(evidence_sentences).lower()

    # Collect recognition/impact signals present in the evidence
    recognition_points = []
    if "critical acclaim" in evidence_text or "critically acclaimed" in evidence_text:
        recognition_points.append("widespread critical acclaim")
    if "grammy" in evidence_text:
        recognition_points.append("Grammy recognition")
    if "ranked" in evidence_text or "greatest albums" in evidence_text or "best albums" in evidence_text:
        recognition_points.append("placement in major greatest-albums rankings")
    if "influential" in evidence_text or "influence" in evidence_text:
        recognition_points.append("lasting influence on the genre")
    if "award" in evidence_text or "prize" in evidence_text:
        recognition_points.append("award recognition")
    if "pioneer" in evidence_text or "pioneering" in evidence_text:
        recognition_points.append("pioneering status in its field")
    if "cultural impact" in evidence_text or "cultural significance" in evidence_text:
        recognition_points.append("significant cultural impact")
    if "commercial" in evidence_text and "success" in evidence_text:
        recognition_points.append("commercial success")

    # Build the subject from the question (extract the main noun phrase)
    subject = "The subject of this question"

    subject_patterns = [
        r"explain why (.+?) is considered important",
        r"explain why (.+?) was considered important",
        r"why is (.+?) important",
        r"why was (.+?) important",
        r"why is (.+?) significant",
        r"why was (.+?) significant",
        r"what makes (.+?) important",
        r"what makes (.+?) significant",
        r"explain the importance of (.+?)(?:[?.]|$)",
    ]

    for pattern in subject_patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            subject = match.group(1).strip().rstrip("?.")
            break

    if not recognition_points:
        # Fall back to a sentence built from the top evidence sentence
        top = evidence_sentences[0].strip()
        if len(top) > 20:
            return f"Based on the retrieved documents: {top}"
        return "The retrieved documents do not provide a clear explanation for this question."

    if len(recognition_points) == 1:
        points_text = recognition_points[0]
    else:
        points_text = ", ".join(recognition_points[:-1]) + " and " + recognition_points[-1]

    return f"{subject} is considered important due to {points_text}, as supported by the retrieved documents."

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    if is_explanation_question(question):
        evidence_sentences = select_evidence_sentences(question, retrieved_chunks)

        evidence = "\n".join(
            f"- {sentence}" for sentence in evidence_sentences
        )

        prompt = f"""
You are answering a music question using only the evidence below.

Write a direct answer in 2 concise sentences.
Do not mention irrelevant chart positions unless they explain importance.
Focus on cultural impact, artistic impact, critical reception, legacy, and influence.

Question:
{question}

Evidence:
{evidence}

Answer:
""".strip()

        return prompt

    context_blocks = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        lines = [
            f"[Document {idx}]",
            f"Title: {chunk['title']}",
            f"Type: {chunk['type']}",
            f"Source: {chunk['source']}",
            f"Text: {chunk['text']}",
        ]

        if chunk.get("artist"):
            lines.insert(3, f"Artist: {chunk['artist']}")

        context_blocks.append("\n".join(lines))

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are answering questions about music using retrieved documents.

Use only the context below.
Do not invent information.
If the answer is a person, album, genre, artist, date, or label, answer directly.
If the question asks for an explanation, answer in 2 or 3 concise sentences.

Question:
{question}

Context:
{context}

Answer:
""".strip()

    return prompt


class AbstractiveQAModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = get_device()

        print(f"Loading generative model: {model_name}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def generate_answer(
        self,
        prompt: str,
        max_input_length: int = 1024,
        max_new_tokens: int = 120,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )

        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return answer.strip()


def answer_question(
    question: str,
    retriever: HybridRetriever,
    generator: AbstractiveQAModel,
    top_k_retrieval: int = 5,
    alpha: float = 0.6,
    filter_type: str | None = None,
    unique_docs: bool = False,
) -> dict:
    retrieved_chunks = retriever.search(
        query=question,
        top_k=top_k_retrieval,
        alpha=alpha,
        unique_docs=unique_docs,
        filter_type=filter_type,
    )

    metadata_answer = metadata_answer_if_possible(question, retrieved_chunks)

    if metadata_answer is not None:
        answer = metadata_answer
        prompt = ""
        answer_method = "metadata"
    else:
        prompt = build_prompt(question, retrieved_chunks)
        answer = generator.generate_answer(prompt)
        answer = clean_generated_answer(answer)
        answer_method = "generative"

        if is_explanation_question(question) and is_bad_explanation_answer(answer, question):
            evidence_sentences = select_evidence_sentences(question, retrieved_chunks)
            answer = synthesize_explanation_from_evidence(question, evidence_sentences)
            answer_method = "evidence_synthesis"

    return {
        "question": question,
        "answer": answer,
        "answer_method": answer_method,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
    }


def print_result(result: dict, show_prompt: bool = False) -> None:
    print("=" * 100)
    print(f"Question: {result['question']}")
    print()
    print("Abstractive answer:")
    print(result["answer"])
    print(f"Answer method: {result.get('answer_method', 'generative')}")

    print()
    print("=" * 100)
    print("Retrieved evidence:")

    for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
        print("-" * 100)
        print(f"{idx}. {chunk['title']} ({chunk['type']})")
        print(f"Score: {chunk['score']:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"URL: {chunk['url']}")
        print()
        print(chunk["text"][:700])

    if show_prompt:
        print()
        print("=" * 100)
        print("Prompt:")
        print(result["prompt"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)

    parser.add_argument("--top-k-retrieval", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--unique-docs", action="store_true")
    parser.add_argument("--show-prompt", action="store_true")

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

    generator = AbstractiveQAModel(GEN_MODEL_NAME)

    result = answer_question(
        question=args.question,
        retriever=retriever,
        generator=generator,
        top_k_retrieval=args.top_k_retrieval,
        alpha=args.alpha,
        filter_type=args.filter_type,
        unique_docs=args.unique_docs,
    )

    print_result(result, show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()