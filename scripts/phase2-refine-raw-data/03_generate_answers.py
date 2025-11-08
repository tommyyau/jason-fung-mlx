#!/usr/bin/env python3
"""
Step 03 – Generate Answers
──────────────────────────
Pairs each question in `data/generated_questions.json` with its source transcript,
asks the LLM for fully formatted answers in Jason Fung's written voice, and streams
results into `data/generated_answers.jsonl`.
"""

import json
import os
import sys
import asyncio
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
possible_env_paths = [
    project_root / ".env",
    project_root / "data" / ".env",
    Path(".env"),
]

env_path = None
for path in possible_env_paths:
    if path.exists():
        env_path = path
        break

if env_path:
    load_dotenv(dotenv_path=env_path)
    print(f"✓ Loaded .env file from: {env_path.resolve()}")
else:
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        print("✓ Loaded .env file from default location")

# ─────────────────────────────
# Config
# ─────────────────────────────
QUESTIONS_FILE = "data/generated_questions.json"
TRANSCRIPTS_FILE = "data/transcripts/transcripts.jsonl"
OUTPUT_FILE = "data/generated_answers.jsonl"
MODEL_LLM = "gpt-5-mini"
MAX_CONCURRENT = 20  # Number of parallel workers

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
client = AsyncOpenAI(api_key=api_key)


# ─────────────────────────────
# Answer Generation
# ─────────────────────────────
async def generate_answers_for_video(
    questions: List[Dict],
    transcript: str,
    video_title: str,
    video_id: str,
    client: AsyncOpenAI,
) -> List[Dict]:
    """
    Generate answers for all questions of a video in one API call.

    Args:
        questions: List of question dicts with 'question', 'tags', 'video_title', 'video_id'
        transcript: Full video transcript text
        video_title: Video title
        video_id: Video ID
        client: AsyncOpenAI client

    Returns:
        List of dicts with: question, answer, video_id, video_title, tags
    """
    # Build questions list for prompt with explicit question numbers
    questions_text = "\n".join(
        [f"Question {i+1}: {q['question']}" for i, q in enumerate(questions)]
    )

    prompt = f"""You are analyzing a complete video transcript from Dr. Jason Fung. Your task is to generate comprehensive, well-formatted answers to specific questions based on this transcript.

Video Title: {video_title}
Video ID: {video_id}

Complete Transcript ({len(transcript):,} characters):
{transcript}

QUESTIONS TO ANSWER:
{questions_text}

CRITICAL REQUIREMENTS:

1. DYNAMIC ANSWER LENGTH:
   - Determine appropriate length based on question complexity and available information
   - Short, simple questions deserve concise answers (50-150 words)
   - Complex questions require detailed explanations (200-500+ words)
   - Answer fully based on what's available in the transcript - don't cut explanations short
   - If the transcript has limited information on a topic, acknowledge that but provide what's available

2. BEAUTIFUL FORMATTING FOR WRITTEN/CHAT INTERFACE:
   - This is NOT conversational speech - format for written responses that people will read
   - Use paragraph breaks (\\n\\n) to separate distinct concepts and improve readability
   - Use **bold text** for key terms, important concepts, and emphasis
   - Use bullet points (-) or numbered lists (1., 2., 3.) when explaining:
     * Multiple items, steps, or related concepts
     * Lists of benefits, causes, effects, or examples
     * Sequential processes or procedures
   - Use line breaks strategically for visual structure
   - Make answers clear, readable, and easy to scan
   - Formatting should enhance understanding, not distract

3. DR. FUNG'S STYLE (ADAPTED FOR WRITTEN FORMAT):
   - Maintain his teaching approach: step-by-step logical explanations
   - Preserve his key analogies (money/mattress, barrel, etc.) - use his exact analogies
   - Keep his practical, accessible tone but adapted for written format
   - Use his exact words/phrases for:
     * Important technical terms and definitions
     * Key analogies and characteristic phrases
     * Specific numbers, percentages, study results, or data points
   - Fix obvious transcription errors (grammatical mistakes, incomplete sentences)
   - Write naturally flowing prose - don't force verbatim quotes that break flow
   - DO NOT use third-person ("Dr. Fung says") - write in direct style
   - CRITICAL: DO NOT reference "the transcript", "the transcript says", "according to the transcript", "the transcript states", "the transcript explains", "the transcript describes", "the transcript emphasizes", "the transcript notes", "the transcript highlights", "the transcript makes clear", "the transcript refers to", "from the transcript", "in the transcript", or any similar phrases
   - Write as if Dr. Fung is directly explaining the concepts - the transcript is just the source material, not something to be referenced in the answer
   - Answers should sound like Dr. Fung's direct explanations, not meta-commentary about a transcript

4. COMPLETE ANSWERS:
   - Answer each question fully based on the transcript
   - Include full context and complete concepts - don't stop mid-explanation
   - If a question can't be fully answered from this transcript, say so and provide what's available
   - Don't add information not present in the transcript

5. FORMATTING EXAMPLES:

GOOD FORMATTING (complex question):
The **energy balance equation** forms the basis of what people call a calorie deficit. It's very simple: body fat equals calories in minus calories out. Calories are simply a unit of energy.

But the very name "energy balance equation" tells us something important: these three variables must always balance. That means there can never truly be a deficit. Think about it this way with a money analogy: if you have $10 in your wallet and want to buy a hamburger that costs $15, you must get the money stored under your mattress to pay.

The key question is: what controls whether you can release the fat energy? The answer is **insulin**. When insulin is high:
- It's a storage hormone that signals the body that food is coming in
- You cannot release body fat energy - you can only store it
- When insulin is low, that's the signal that you can now release this stored energy

GOOD FORMATTING (simple question):
**Insulin** is a storage hormone. When insulin levels are high, you cannot release energy from body fat - you can only store it. When insulin is low, you can release stored energy.

BAD FORMATTING (conversational, not formatted):
"Well, you know, the energy balance equation is the basis. Very name is the energy balance equation which means that these are three variables must always balance." (WRONG - transcription errors, no formatting, poor flow)

BAD FORMATTING (references transcript):
"According to the transcript, refined carbohydrates really stimulate a lot of insulin. The transcript states that..." (WRONG - never reference the transcript in answers)

OUTPUT FORMAT:
Return ONLY a JSON array of objects. Each object must have this exact structure:
[
  {{
    "question_number": 1,
    "answer": "Comprehensive, well-formatted answer with proper formatting (paragraphs, bold, lists as appropriate). Answer length should match question complexity. Use \\n\\n for paragraph breaks, **text** for bold, and bullet points or numbered lists when appropriate."
  }},
  {{
    "question_number": 2,
    "answer": "..."
  }},
  ...
]

CRITICAL REQUIREMENTS:
- Use "question_number" (integer 1, 2, 3, etc.) to identify which question you're answering
- The question_number MUST match the question number from the list above (Question 1, Question 2, etc.)
- You MUST answer all questions (1 through {len(questions)})
- Return answers in order (question_number 1, then 2, then 3, etc.)
- Do NOT include the question text in your response - only the question_number and answer

IMPORTANT:
- Answer ALL questions from the list above
- Use dynamic length - short questions get short answers, complex questions get detailed answers
- Format beautifully for readability
- Maintain Dr. Fung's style but adapted for written format
- Return valid JSON only, no markdown code blocks
"""

    # Try up to 3 times to get valid JSON
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=MODEL_LLM,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16000,  # Allow longer responses for multiple answers
            )
            content = resp.choices[0].message.content.strip()

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse JSON
            result = json.loads(content)

            # Ensure it's a list
            if not isinstance(result, list):
                if isinstance(result, dict):
                    result = [result]
                else:
                    return []

            # Match answers to questions
            # CRITICAL: Only match by question text - NEVER use order-based matching
            # If we can't match by text, we SKIP that question-answer pair (better to skip than guess wrong)

            results = []

            # Extract answers by question number (RELIABLE - no text matching needed!)
            answers_by_number = {}  # question_number -> answer
            answers_list = []  # Keep for validation
            for item in result:
                # Support both old format (with "question" text) and new format (with "question_number")
                if "question_number" in item and "answer" in item:
                    q_num = item.get("question_number")
                    a_text = item.get("answer", "").strip()
                    if isinstance(q_num, int) and a_text:
                        answers_by_number[q_num] = a_text
                        answers_list.append((q_num, a_text))
                elif "question" in item and "answer" in item:
                    # Fallback for old format - store for text-based matching
                    q_text = item.get("question", "").strip()
                    a_text = item.get("answer", "").strip()
                    if q_text and a_text:
                        answers_list.append((q_text, a_text))

            # Match by question number (RELIABLE - no text matching needed!)
            results = []
            unmatched_questions = []

            # If we have question_number format, use it (most reliable)
            if answers_by_number:
                for i, q in enumerate(questions):
                    question_number = i + 1
                    question_text = q.get("question", "").strip()

                    if question_number in answers_by_number:
                        answer = answers_by_number[question_number]
                        results.append(
                            {
                                "question": question_text,
                                "answer": answer,
                                "video_id": video_id,
                                "video_title": video_title,
                                "tags": q.get("tags", []),
                            }
                        )
                    else:
                        unmatched_questions.append((question_number, question_text))
                        print(
                            f"     ❌ Question {question_number}: No answer returned for this question number",
                            flush=True,
                        )
                        print(f"        Question: {question_text[:80]}...", flush=True)

                # Check for extra answers (question numbers that don't exist)
                max_question_num = len(questions)
                extra_answers = [
                    num
                    for num in answers_by_number.keys()
                    if num > max_question_num or num < 1
                ]
                if extra_answers:
                    print(
                        f"     ⚠️  WARNING: Got answers for invalid question numbers: {extra_answers}",
                        flush=True,
                    )

                # Summary
                matched_count = len(results)
                total_questions = len(questions)
                if matched_count == total_questions:
                    print(
                        f"     ✓ All {matched_count} questions matched by question number!",
                        flush=True,
                    )
                else:
                    print(
                        f"     📊 Matched {matched_count}/{total_questions} questions by question number",
                        flush=True,
                    )
                    if unmatched_questions:
                        unmatched_numbers = [num for num, _ in unmatched_questions]
                        print(
                            f"        Unmatched question numbers: {unmatched_numbers}",
                            flush=True,
                        )

                return results

            # FALLBACK: If no question_number format, use old text-based matching
            # (This should rarely happen with the new prompt, but kept for backward compatibility)
            if not answers_by_number and answers_list:
                # Build text lookup for fallback
                answers_by_question_text = {}
                for item in answers_list:
                    if isinstance(item, tuple) and len(item) == 2:
                        q_text, a_text = item
                        if isinstance(q_text, str) and a_text:
                            answers_by_question_text[q_text] = a_text
            else:
                answers_by_question_text = {}

            matched_question_texts = set()

            def normalize_for_matching(text: str) -> str:
                """Normalize text for more flexible matching."""
                import unicodedata
                import re

                # Normalize unicode (e.g., smart quotes to regular quotes)
                text = unicodedata.normalize("NFKD", text)
                # Convert to lowercase
                text = text.lower()
                # Normalize whitespace
                text = " ".join(text.split())
                # Replace various quote types with standard quotes
                text = text.replace('"', '"').replace('"', '"').replace("’", "'").replace("‘", "'")
                # Replace various dash types with regular dash
                text = text.replace("—", "-").replace("–", "-").replace("−", "-")
                # Normalize brackets and parentheses
                text = text.replace("[", "(").replace("]", ")")
                text = text.replace("{", "(").replace("}", ")")
                return text.strip()

            def calculate_similarity(text1: str, text2: str) -> float:
                """Calculate simple similarity ratio between two texts."""
                if not text1 or not text2:
                    return 0.0
                # Simple character-based similarity
                longer = text1 if len(text1) > len(text2) else text2
                shorter = text2 if longer == text1 else text1
                matches = sum(1 for a, b in zip(longer, shorter) if a == b)
                return matches / len(longer) if longer else 0.0

            # Match each question STRICTLY by text only
            for i, q in enumerate(questions):
                question_text = q.get("question", "").strip()
                answer = None
                match_found = False
                match_attempts = {
                    "exact": False,
                    "case_insensitive": False,
                    "whitespace_normalized": False,
                    "punctuation_normalized": False,
                    "fuzzy_high_similarity": False,
                }
                best_match_candidate = None
                best_similarity = 0.0

                # Try exact match first (most reliable)
                if question_text in answers_by_question_text:
                    match_attempts["exact"] = True
                    if question_text not in matched_question_texts:
                        answer = answers_by_question_text[question_text]
                        matched_question_texts.add(question_text)
                        match_found = True
                    else:
                        print(
                            f"     ❌ Question {i+1}: Answer already matched to another question (duplicate)",
                            flush=True,
                        )
                        print(f"        Question: {question_text[:80]}...", flush=True)

                # Try case-insensitive match (only if exact failed)
                if not match_found:
                    match_attempts["case_insensitive"] = True
                    question_lower = question_text.lower()
                    for q_key, a_value in answers_by_question_text.items():
                        if q_key.lower() == question_lower and q_key not in matched_question_texts:
                            answer = a_value
                            matched_question_texts.add(q_key)
                            match_found = True
                            print(
                                f"     ⚠️  Question {i+1}: Matched by case-insensitive text (exact match failed)",
                                flush=True,
                            )
                            print(f"        Expected: {question_text[:60]}...", flush=True)
                            print(f"        Matched:  {q_key[:60]}...", flush=True)
                            break

                # Try whitespace-normalized match (only if both above failed)
                if not match_found:
                    match_attempts["whitespace_normalized"] = True
                    question_normalized = " ".join(question_text.split())
                    for q_key, a_value in answers_by_question_text.items():
                        q_key_normalized = " ".join(q_key.split())
                        if (
                            q_key_normalized.lower() == question_normalized.lower()
                            and q_key not in matched_question_texts
                        ):
                            answer = a_value
                            matched_question_texts.add(q_key)
                            match_found = True
                            print(
                                f"     ⚠️  Question {i+1}: Matched by normalized whitespace (exact/case-insensitive failed)",
                                flush=True,
                            )
                            print(f"        Expected: {question_text[:60]}...", flush=True)
                            print(f"        Matched:  {q_key[:60]}...", flush=True)
                            break

                # Try punctuation-normalized match (handles quotes, dashes, brackets, etc.)
                if not match_found:
                    match_attempts["punctuation_normalized"] = True
                    question_norm = normalize_for_matching(question_text)
                    for q_key, a_value in answers_by_question_text.items():
                        q_key_norm = normalize_for_matching(q_key)
                        if q_key_norm == question_norm and q_key not in matched_question_texts:
                            answer = a_value
                            matched_question_texts.add(q_key)
                            match_found = True
                            print(
                                f"     ⚠️  Question {i+1}: Matched by punctuation-normalized text",
                                flush=True,
                            )
                            print(f"        Expected: {question_text[:60]}...", flush=True)
                            print(f"        Matched:  {q_key[:60]}...", flush=True)
                            break

                # Try fuzzy matching for high similarity (>= 95% similar) as last resort
                if not match_found:
                    match_attempts["fuzzy_high_similarity"] = True
                    question_norm = normalize_for_matching(question_text)
                    for q_key, a_value in answers_by_question_text.items():
                        if q_key in matched_question_texts:
                            continue
                        q_key_norm = normalize_for_matching(q_key)
                        # Calculate similarity on normalized text
                        similarity = calculate_similarity(question_norm, q_key_norm)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_candidate = (q_key, a_value)

                    # Only use fuzzy match if similarity is very high (>= 95%)
                    if best_match_candidate and best_similarity >= 0.95:
                        answer = best_match_candidate[1]
                        matched_question_texts.add(best_match_candidate[0])
                        match_found = True
                        print(
                            f"     ⚠️  Question {i+1}: Matched by fuzzy similarity ({best_similarity*100:.1f}% similar)",
                            flush=True,
                        )
                        print(f"        Expected: {question_text[:60]}...", flush=True)
                        print(f"        Matched:  {best_match_candidate[0][:60]}...", flush=True)

                # CRITICAL: If we can't match by text, we SKIP it (never use order-based matching)
                if match_found and answer:
                    results.append(
                        {
                            "question": question_text,
                            "answer": answer,
                            "video_id": video_id,
                            "video_title": video_title,
                            "tags": q.get("tags", []),
                        }
                    )
                else:
                    unmatched_questions.append((i + 1, question_text))
                    print(
                        f"     ❌ Question {i+1}: NO MATCH FOUND - SKIPPING (better to skip than match wrong)",
                        flush=True,
                    )
                    print(
                        f"        ════════════════════════════════════════════════════════════════",
                        flush=True,
                    )
                    print(f"        EXPECTED QUESTION (what we sent):", flush=True)
                    print(f"        {question_text}", flush=True)
                    print(
                        f"        ════════════════════════════════════════════════════════════════",
                        flush=True,
                    )

                    # Show all the questions we got back from LLM for comparison
                    print(f"        QUESTIONS RETURNED BY LLM (for comparison):", flush=True)
                    if answers_by_question_text:
                        for j, (returned_q, _) in enumerate(answers_list, 1):
                            print(f"        [{j}] {returned_q}", flush=True)
                    else:
                        print(f"        (No questions returned in response)", flush=True)

                    print(
                        f"        ════════════════════════════════════════════════════════════════",
                        flush=True)
                    print(f"        MATCHING ATTEMPTS MADE:", flush=True)
                    if match_attempts["exact"]:
                        print(f"        ✗ Exact character match: FAILED", flush=True)
                    if match_attempts["case_insensitive"]:
                        print(f"        ✗ Case-insensitive match: FAILED", flush=True)
                    if match_attempts["whitespace_normalized"]:
                        print(f"        ✗ Whitespace-normalized match: FAILED", flush=True)
                    if match_attempts["punctuation_normalized"]:
                        print(f"        ✗ Punctuation-normalized match: FAILED", flush=True)
                    if match_attempts["fuzzy_high_similarity"]:
                        print(f"        ✗ Fuzzy similarity match (>=95%): FAILED", flush=True)
                        if best_match_candidate:
                            print(
                                f"        Best similarity found: {best_similarity*100:.1f}% (need >=95%)",
                                flush=True,
                            )
                            print(
                                f"        Best candidate: {best_match_candidate[0][:60]}...",
                                flush=True,
                            )

                    # Show character-by-character differences if we have a close match
                    if best_match_candidate and best_similarity >= 0.80:
                        print(
                            f"        ════════════════════════════════════════════════════════════════",
                            flush=True,
                        )
                        print(f"        CHARACTER DIFFERENCES (best candidate):", flush=True)
                        expected_norm = normalize_for_matching(question_text)
                        got_norm = normalize_for_matching(best_match_candidate[0])
                        # Show where they differ
                        diff_positions = []
                        min_len = min(len(expected_norm), len(got_norm))
                        for pos in range(min_len):
                            if expected_norm[pos] != got_norm[pos]:
                                diff_positions.append(pos)
                        if diff_positions:
                            print(
                                f"        Differences at positions: {diff_positions[:10]}",
                                flush=True,
                            )
                            if diff_positions:
                                first_diff = diff_positions[0]
                                start = max(0, first_diff - 20)
                                end = min(len(expected_norm), first_diff + 20)
                                print(
                                    f"        Expected context: ...{expected_norm[start:end]}...",
                                    flush=True,
                                )
                                print(
                                    f"        Got context:      ...{got_norm[start:end]}...",
                                    flush=True,
                                )

                    print(
                        f"        ════════════════════════════════════════════════════════════════",
                        flush=True,
                    )
                    print(
                        f"        ⚠️  This question will NOT be included in results to prevent wrong pairing",
                        flush=True,
                    )
                    print(f"", flush=True)

            # Validation and reporting
            matched_count = len(results)
            total_questions = len(questions)
            unmatched_count = len(unmatched_questions)

            # Summary per video
            print(f"     📊 MATCHING SUMMARY for this video:", flush=True)
            print(
                f"        ✓ Matched: {matched_count}/{total_questions} questions", flush=True
            )
            if unmatched_count > 0:
                print(
                    f"        ✗ Unmatched: {unmatched_count}/{total_questions} questions",
                    flush=True,
                )
                unmatched_numbers = [num for num, _ in unmatched_questions]
                print(
                    f"        Unmatched question numbers: {unmatched_numbers}",
                    flush=True,
                )
            else:
                print(f"        ✓ All questions matched successfully!", flush=True)

            if unmatched_questions:
                print(
                    f"     ⚠️  WARNING: {unmatched_count} question(s) from THIS VIDEO could not be matched and were SKIPPED",
                    flush=True,
                )
                print(
                    f"        This prevents wrong answer-question pairings in training data",
                    flush=True,
                )

            # Check for unmatched answers (answers returned but no question matched)
            unmatched_answers = [
                q_text
                for q_text in answers_by_question_text.keys()
                if q_text not in matched_question_texts
            ]
            if unmatched_answers:
                print(
                    f"     ⚠️  WARNING: {len(unmatched_answers)} answer(s) returned but no matching question found",
                    flush=True,
                )
                for q_text in unmatched_answers[:3]:
                    print(f"        Unmatched answer for: {q_text[:60]}...", flush=True)

            # Validate order as sanity check (but don't use it for matching)
            if len(answers_list) == len(questions):
                order_matches = True
                for i, (returned_q_text, _) in enumerate(answers_list):
                    expected_q_text = questions[i].get("question", "").strip()
                    if returned_q_text.lower() != expected_q_text.lower():
                        order_matches = False
                        break
                if not order_matches:
                    print(
                        f"     ⚠️  Note: LLM returned answers OUT OF ORDER (but matched correctly by text)",
                        flush=True,
                    )
            elif len(answers_list) != len(questions):
                print(
                    f"     ⚠️  Note: Got {len(answers_list)} answers but expected {len(questions)} questions",
                    flush=True,
                )

            return results

        except json.JSONDecodeError as e:
            if attempt < 2:  # Retry on JSON errors
                print(f"     ⚠️  JSON parse error (attempt {attempt + 1}/3), retrying...", flush=True)
                await asyncio.sleep(2)  # Brief pause before retry
                continue
            else:
                print(f"     ⚠️  JSON parse error after 3 attempts: {str(e)[:100]}", flush=True)
                print(f"     Response preview: {content[:500] if 'content' in locals() else 'N/A'}", flush=True)
                print(f"     Video: {video_title} ({video_id}), Questions: {len(questions)}", flush=True)
                return []

        except Exception as e:
            if attempt < 2:
                print(f"     ⚠️  Error (attempt {attempt + 1}/3): {type(e).__name__}, retrying...", flush=True)
                await asyncio.sleep(2)
                continue
            else:
                error_msg = str(e)
                print(f"     ⚠️  Error generating answers after 3 attempts: {type(e).__name__}: {error_msg[:200]}", flush=True)
                print(f"     Video: {video_title} ({video_id}), Questions: {len(questions)}", flush=True)
                return []

    return []


# ─────────────────────────────
# Async Video Processing
# ─────────────────────────────
async def process_video_async(
    video_id: str,
    questions: List[Dict],
    transcript: str,
    video_title: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    video_index: int,
    total_videos: int,
    output_file: Path,
) -> int:
    """
    Process a single video asynchronously with semaphore for concurrency control.
    Writes results incrementally to output file.

    Returns:
        Number of answers generated
    """
    async with semaphore:  # Limit concurrent processing
        print(
            f"  [{video_index}/{total_videos}] Processing: {video_title[:60]}... ({len(questions)} questions)",
            flush=True,
        )

        # Show which questions are being answered
        question_preview = ", ".join(
            [
                q.get("question", "")[:50] + "..." if len(q.get("question", "")) > 50 else q.get("question", "")
                for q in questions[:3]
            ]
        )
        if len(questions) > 3:
            question_preview += f" ... and {len(questions) - 3} more"
        print(f"     → Answering questions: {question_preview}", flush=True)
        print(f"     → Calling API...", flush=True)

        video_start = time.time()

        try:
            answers = await generate_answers_for_video(
                questions, transcript, video_title, video_id, client
            )
            video_time = time.time() - video_start

            # Write results incrementally
            if answers:
                with open(output_file, "a", encoding="utf-8") as f:
                    for answer in answers:
                        json_line = json.dumps(answer, ensure_ascii=False) + "\n"
                        f.write(json_line)
                        f.flush()

                matched_count = len(answers)
                total_questions = len(questions)
                if matched_count == total_questions:
                    print(
                        f"     ✓ Generated {matched_count}/{total_questions} answers - ALL MATCHED! ({video_time:.1f}s)",
                        flush=True,
                    )
                else:
                    print(
                        f"     ✓ Generated {matched_count}/{total_questions} answers ({video_time:.1f}s)",
                        flush=True,
                    )
                return len(answers)
            else:
                print(f"     ⚠️  No answers generated ({video_time:.1f}s)", flush=True)
                return 0

        except Exception as e:
            print(f"     ⚠️  Error processing video: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return 0


# ─────────────────────────────
# Main Processing
# ─────────────────────────────
async def main_async():
    """Main async function to process all videos in parallel."""
    # Load questions
    questions_path = project_root / QUESTIONS_FILE
    if not questions_path.exists():
        print(f"❌ Questions file not found: {questions_path}")
        return

    print(f"→ Loading questions from {QUESTIONS_FILE}...")
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in questions file: {str(e)[:200]}")
        print(f"   Please run: python3 newscripts/02_validate_questions.py")
        return
    except Exception as e:
        print(f"❌ Error loading questions file: {str(e)}")
        return

    # Validate structure
    if not isinstance(all_questions, list):
        print(f"❌ Questions file must contain a JSON array, got {type(all_questions).__name__}")
        return

    # Quick validation of required fields
    required_fields = ["video_title", "video_id", "question", "tags"]
    invalid_questions = []
    for i, q in enumerate(all_questions):
        if not isinstance(q, dict):
            invalid_questions.append(f"Question {i+1}: Not a dictionary")
            continue
        for field in required_fields:
            if field not in q:
                invalid_questions.append(f"Question {i+1}: Missing '{field}'")
                break

    if invalid_questions:
        print(f"❌ Found {len(invalid_questions)} invalid question(s):")
        for issue in invalid_questions[:10]:
            print(f"   • {issue}")
        if len(invalid_questions) > 10:
            print(f"   ... and {len(invalid_questions) - 10} more")
        print(f"   Please run: python3 newscripts/02_validate_questions.py")
        return

    print(f"→ Loaded {len(all_questions)} questions")

    # Group questions by video_id
    questions_by_video = defaultdict(list)
    for q in all_questions:
        video_id = q.get("video_id")
        if video_id:
            questions_by_video[video_id].append(q)

    print(f"→ Grouped into {len(questions_by_video)} videos")

    # Load transcripts
    transcripts_path = project_root / TRANSCRIPTS_FILE
    if not transcripts_path.exists():
        print(f"❌ Transcripts file not found: {transcripts_path}")
        return

    print(f"→ Loading transcripts from {TRANSCRIPTS_FILE}...")
    transcripts = {}
    with open(transcripts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    video_id = data.get("video_id")
                    transcript = data.get("transcript", "")
                    title = data.get("title", "Untitled")
                    if video_id and transcript:
                        transcripts[video_id] = {
                            "transcript": transcript,
                            "title": title,
                        }
                except json.JSONDecodeError:
                    continue

    print(f"→ Loaded {len(transcripts)} transcripts")

    # Filter to videos that have both questions and transcripts
    videos_to_process = []
    for video_id, questions in questions_by_video.items():
        if video_id in transcripts:
            videos_to_process.append(
                {
                    "video_id": video_id,
                    "questions": questions,
                    "transcript": transcripts[video_id]["transcript"],
                    "title": transcripts[video_id]["title"],
                }
            )
        else:
            print(f"  ⚠️  No transcript found for video_id: {video_id} ({len(questions)} questions)")

    if not videos_to_process:
        print("❌ No videos to process (no matching transcripts found)")
        return

    print(f"→ Processing {len(videos_to_process)} videos with {MAX_CONCURRENT} parallel workers")
    total_questions = sum(len(v["questions"]) for v in videos_to_process)
    print(f"→ Generating answers for {total_questions} total questions using Dr. Fung's style...\n")

    # Prepare output file
    output_path = project_root / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear or create output file
    with open(output_path, "w", encoding="utf-8") as f:
        pass  # Create empty file

    start_time = time.time()

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Process all videos in parallel
    tasks = [
        process_video_async(
            video["video_id"],
            video["questions"],
            video["transcript"],
            video["title"],
            client,
            semaphore,
            i + 1,
            len(videos_to_process),
            output_path,
        )
        for i, video in enumerate(videos_to_process)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count total answers and questions
    total_answers = 0
    total_questions_expected = sum(len(v["questions"]) for v in videos_to_process)
    failed_videos = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  ⚠️  Video {i+1} failed with exception: {type(result).__name__}")
            failed_videos += 1
        elif isinstance(result, int):
            total_answers += result

    total_time = time.time() - start_time
    total_unmatched = total_questions_expected - total_answers
    match_rate = (total_answers / total_questions_expected * 100) if total_questions_expected > 0 else 0

    print(f"\n{'='*70}")
    print(f"✅ FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"   Videos processed: {len(videos_to_process)}")
    print(f"   Videos failed: {failed_videos}")
    print(f"   Total questions: {total_questions_expected}")
    print(f"   Answers generated: {total_answers}")
    if total_unmatched > 0:
        print(f"   Questions unmatched: {total_unmatched} ({100-match_rate:.1f}%)")
    print(f"   Match rate: {match_rate:.1f}%")
    print(f"   Time: {total_time/60:.1f} minutes ({total_time/len(videos_to_process):.1f}s per video)")
    print(f"   Output: {output_path}")
    print(f"{'='*70}")


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


