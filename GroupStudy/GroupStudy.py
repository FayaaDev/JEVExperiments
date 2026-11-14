#!/usr/bin/env python3

"""
WhatsApp AI/vibe-coding interaction classifier.

Reads a CSV file, classifies every row using the ZTEXT column, and writes all
rows back out with additional classification columns.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Iterable, Literal

from agents import Agent, RunConfig, Runner
from dotenv import load_dotenv
from openai import AuthenticationError
from pydantic import BaseModel, Field


load_dotenv(override=True)

RUN_CONFIG = RunConfig(tracing_disabled=True)


DEFAULT_INPUT_CSV = "data.csv"
DEFAULT_OUTPUT_CSV = "useful_interactions.csv"
DEFAULT_CACHE_FILE = "whatsapp_interaction_cache.json"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_BATCH_SIZE = 25
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
PROMPT_VERSION = "2026-06-29-v1"

OUTPUT_COLUMNS = [
    "is_useful_interaction",
    "interaction_category",
    "classification_confidence",
    "classification_reason",
    "needs_review",
]


class BatchMessage(BaseModel):
    """Single message to classify."""

    message_id: str = Field(description="Stable identifier for this message within the batch.")
    text: str = Field(description="Original WhatsApp message from the ZTEXT column.")


class MessageClassification(BaseModel):
    """Classification returned by the agent for one message."""

    message_id: str
    is_useful_interaction: bool
    interaction_category: Literal[
        "ai_discussion",
        "vibe_coding",
        "ai_coding",
        "resource_share",
        "not_relevant",
        "uncertain",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    needs_review: bool


class BatchClassificationResult(BaseModel):
    """Structured result for a batch classification call."""

    results: list[MessageClassification]


@dataclass
class ClassificationContext:
    """Runtime metadata for a classification run."""

    cache_hits: int = 0
    cache_misses: int = 0
    batches_processed: int = 0
    rows_written: int = 0


def create_classifier_agent(model: str) -> Agent[ClassificationContext]:
    """Create the WhatsApp classifier agent."""

    instructions = """
You classify WhatsApp messages from a CSV column named ZTEXT.

Your job is to identify whether each message meaningfully contributes to AI and
vibe-coding discussion.

Useful interactions include messages that:
- discuss AI tools, models, prompts, agents, MCP, SDKs, APIs, automation, or
  coding workflows
- ask or answer technical questions about building, debugging, integrating, or
  shipping software with AI assistance
- share a link or resource when the message itself makes its AI/coding
  relevance clear
- describe experiments, decisions, problems, or results related to AI-assisted
  software development or vibe coding

Not useful interactions include messages that are mainly:
- greetings, thanks, blessings, congratulations, or holiday wishes
- short acknowledgements, praise, laughter, reactions, or emojis
- general social chat unrelated to AI or coding
- bare links without enough context to prove relevance
- ambiguous snippets where relevance cannot be inferred confidently

Important rules:
- Use semantics, not fixed keywords.
- Arabic, English, and mixed Arabic/English are all valid.
- Be conservative. If uncertain, set interaction_category to "uncertain" and
  needs_review to true.
- Return exactly one result per input message.
- Keep the reason short and specific.
- Return structured data only.
""".strip()

    return Agent[ClassificationContext](
        name="WhatsApp AI Interaction Classifier",
        instructions=instructions,
        output_type=BatchClassificationResult,
        model=model,
    )


def normalize_text_for_cache(text: str) -> str:
    """Normalize whitespace so identical messages share cache entries."""

    return (text or "").strip()


def message_hash(text: str) -> str:
    """Create a stable hash for a message."""

    normalized = normalize_text_for_cache(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_key(text_hash: str, model: str, base_url: str | None) -> str:
    """Build a cache key that invalidates when prompt or model changes."""

    provider = (base_url or "default").rstrip("/")
    return f"{PROMPT_VERSION}:{provider}:{model}:{text_hash}"


def load_cache(cache_path: Path) -> Dict[str, dict]:
    """Load cached classifications if available."""

    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("r", encoding="utf-8") as infile:
            data = json.load(infile)
    except JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Cache file must contain a JSON object: {cache_path}")

    return data


def save_cache(cache_path: Path, cache: Dict[str, dict]) -> None:
    """Persist cached classifications."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as outfile:
        json.dump(cache, outfile, ensure_ascii=False, indent=2)


def chunked(items: list[BatchMessage], size: int) -> Iterable[list[BatchMessage]]:
    """Yield batches of messages."""

    for index in range(0, len(items), size):
        yield items[index:index + size]


def build_batch_prompt(messages: list[BatchMessage]) -> str:
    """Serialize a classification batch for the agent."""

    payload = {
        "messages": [message.model_dump() for message in messages],
        "task": (
            "Classify each message for whether it is a useful AI or vibe-coding "
            "interaction. Return exactly one result for each message_id."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def validate_batch_result(
    messages: list[BatchMessage],
    result: BatchClassificationResult,
) -> dict[str, MessageClassification]:
    """Ensure the agent returned one valid result for each message."""

    expected_ids = {message.message_id for message in messages}
    returned_ids = [item.message_id for item in result.results]

    if len(returned_ids) != len(expected_ids):
        raise ValueError(
            "Classifier returned the wrong number of results for a batch: "
            f"expected {len(expected_ids)}, got {len(returned_ids)}"
        )

    if len(set(returned_ids)) != len(returned_ids):
        raise ValueError("Classifier returned duplicate message_id values in a batch")

    if set(returned_ids) != expected_ids:
        missing = sorted(expected_ids - set(returned_ids))
        extra = sorted(set(returned_ids) - expected_ids)
        raise ValueError(
            "Classifier returned mismatched message_id values. "
            f"Missing: {missing}. Extra: {extra}."
        )

    return {item.message_id: item for item in result.results}


def read_input_rows(input_path: Path, limit: int | None) -> tuple[list[dict], list[str]]:
    """Read the input CSV and validate required columns."""

    with input_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        if "ZTEXT" not in fieldnames:
            raise ValueError("Input CSV must contain a ZTEXT column")

        rows = list(reader)

    if limit is not None:
        rows = rows[:limit]

    return rows, fieldnames


async def classify_unique_messages(
    *,
    rows: list[dict],
    model: str,
    base_url: str | None,
    batch_size: int,
    cache_path: Path,
) -> tuple[Dict[str, MessageClassification], ClassificationContext]:
    """Classify each unique message, reusing cached results when possible."""

    context = ClassificationContext()
    cache = load_cache(cache_path)
    classifications: Dict[str, MessageClassification] = {}
    pending_messages: list[BatchMessage] = []
    cached_hashes: set[str] = set()
    pending_hashes: set[str] = set()

    for row in rows:
        text = row.get("ZTEXT", "") or ""
        text_hash = message_hash(text)
        key = cache_key(text_hash, model, base_url)

        cached = cache.get(key)
        if cached is not None:
            try:
                classifications[text_hash] = MessageClassification.model_validate(cached)
            except Exception:
                cache.pop(key, None)
                cached = None

        if cached is not None:
            if text_hash not in cached_hashes:
                cached_hashes.add(text_hash)
                context.cache_hits += 1
            continue

        if text_hash in classifications or text_hash in pending_hashes:
            continue

        pending_messages.append(
            BatchMessage(
                message_id=text_hash,
                text=text.strip(),
            )
        )
        pending_hashes.add(text_hash)
        context.cache_misses += 1

    if not pending_messages:
        return classifications, context

    agent = create_classifier_agent(model)

    for batch in chunked(pending_messages, batch_size):
        prompt = build_batch_prompt(batch)
        run_result = await Runner.run(
            starting_agent=agent,
            input=prompt,
            context=context,
            run_config=RUN_CONFIG,
            max_turns=6,
        )

        if run_result.final_output is None:
            raise ValueError("Classifier agent did not return structured output")

        batch_results = validate_batch_result(batch, run_result.final_output)

        for text_hash, classification in batch_results.items():
            classifications[text_hash] = classification
            cache[cache_key(text_hash, model, base_url)] = classification.model_dump(mode="json")

        save_cache(cache_path, cache)
        context.batches_processed += 1

    return classifications, context


def write_output_rows(
    *,
    rows: list[dict],
    fieldnames: list[str],
    classifications: Dict[str, MessageClassification],
    output_path: Path,
) -> None:
    """Write all original rows with classification columns appended."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = fieldnames + [column for column in OUTPUT_COLUMNS if column not in fieldnames]

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fields)
        writer.writeheader()

        for row in rows:
            text_hash = message_hash(row.get("ZTEXT", "") or "")
            classification = classifications[text_hash]
            row["is_useful_interaction"] = "true" if classification.is_useful_interaction else "false"
            row["interaction_category"] = classification.interaction_category
            row["classification_confidence"] = f"{classification.confidence:.2f}"
            row["classification_reason"] = classification.reason
            row["needs_review"] = "true" if classification.needs_review else "false"
            writer.writerow(row)


async def process_csv(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    base_url: str | None,
    batch_size: int,
    cache_path: Path,
    limit: int | None,
) -> None:
    """Run the classification pipeline and print a summary."""

    rows, fieldnames = read_input_rows(input_path, limit)
    classifications, context = await classify_unique_messages(
        rows=rows,
        model=model,
        base_url=base_url,
        batch_size=batch_size,
        cache_path=cache_path,
    )
    write_output_rows(
        rows=rows,
        fieldnames=fieldnames,
        classifications=classifications,
        output_path=output_path,
    )
    context.rows_written = len(rows)

    useful_rows = 0
    review_rows = 0
    categories: dict[str, int] = {}

    for row in rows:
        classification = classifications[message_hash(row.get("ZTEXT", "") or "")]
        if classification.is_useful_interaction:
            useful_rows += 1
        if classification.needs_review:
            review_rows += 1
        categories[classification.interaction_category] = categories.get(classification.interaction_category, 0) + 1

    print(f"Wrote: {output_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Useful interactions: {useful_rows}")
    print(f"Needs review: {review_rows}")
    print(f"Cache hits: {context.cache_hits}")
    print(f"Unique messages sent to model: {context.cache_misses}")
    print(f"Batches processed: {context.batches_processed}")
    if categories:
        print("Categories:")
        for category, count in sorted(categories.items(), key=lambda item: (-item[1], item[0])):
            print(f"  - {category}: {count}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Classify WhatsApp CSV rows by whether ZTEXT contributes to AI and vibe-coding discussion."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=DEFAULT_INPUT_CSV,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default=DEFAULT_OUTPUT_CSV,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Optional OpenAI-compatible base URL, for example http://127.0.0.1:8317/v1.",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=(
            "Environment variable name containing the API key to use. "
            f"Default: {DEFAULT_API_KEY_ENV}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of unique messages per model call. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--cache-file",
        default=DEFAULT_CACHE_FILE,
        help=f"JSON cache file for deduplicated message classifications. Default: {DEFAULT_CACHE_FILE}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of CSV rows to process.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0 when provided")

    return args


async def main() -> None:
    """Main async entry point."""

    args = parse_args()
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    selected_api_key = os.getenv(args.api_key_env)
    if not selected_api_key:
        raise SystemExit(f"Missing API key in ${args.api_key_env}.")

    os.environ["OPENAI_API_KEY"] = selected_api_key

    try:
        await process_csv(
            Path(args.input_csv),
            Path(args.output_csv),
            model=args.model,
            base_url=args.base_url,
            batch_size=args.batch_size,
            cache_path=Path(args.cache_file),
            limit=args.limit,
        )
    except AuthenticationError as error:
        raise SystemExit(
            "OpenAI authentication failed. Verify the selected base URL and the API key "
            f"from ${args.api_key_env}."
        ) from error


if __name__ == "__main__":
    asyncio.run(main())
