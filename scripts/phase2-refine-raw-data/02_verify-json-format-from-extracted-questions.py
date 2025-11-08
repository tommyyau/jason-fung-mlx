#!/usr/bin/env python3
"""
Step 02 – Validate Questions (Optional)
───────────────────────────────────────
Checks `data/generated_questions.json` for JSON compliance, required fields, and basic integrity
before answer generation. Can auto-fix common issues like trailing commas and malformed tags.
"""

import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Load paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
QUESTIONS_FILE = project_root / "data" / "generated_questions.json"


def validate_questions_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate the generated questions JSON file.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check if file exists
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]

    # Try to parse as JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {str(e)}"]

    # Try to parse JSON
    try:
        questions = json.loads(content)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {str(e)}"]

    # Check if it's a list
    if not isinstance(questions, list):
        return False, [f"Root element must be a list, got {type(questions).__name__}"]

    # Validate each question
    required_fields = ["video_title", "video_id", "question", "tags"]
    for i, question in enumerate(questions):
        if not isinstance(question, dict):
            issues.append(f"Question {i+1}: Not a dictionary, got {type(question).__name__}")
            continue

        # Check required fields
        for field in required_fields:
            if field not in question:
                issues.append(f"Question {i+1}: Missing required field '{field}'")

        # Validate field types
        if "video_title" in question and not isinstance(question["video_title"], str):
            issues.append(f"Question {i+1}: 'video_title' must be a string")

        if "video_id" in question and not isinstance(question["video_id"], str):
            issues.append(f"Question {i+1}: 'video_id' must be a string")

        if "question" in question:
            if not isinstance(question["question"], str):
                issues.append(f"Question {i+1}: 'question' must be a string")
            elif not question["question"].strip():
                issues.append(f"Question {i+1}: 'question' is empty")

        if "tags" in question:
            if not isinstance(question["tags"], list):
                issues.append(f"Question {i+1}: 'tags' must be a list")
            else:
                for j, tag in enumerate(question["tags"]):
                    if not isinstance(tag, str):
                        issues.append(f"Question {i+1}, tag {j+1}: Tag must be a string, got {type(tag).__name__}")
                    elif not tag.strip():
                        issues.append(f"Question {i+1}, tag {j+1}: Tag is empty")

    is_valid = len(issues) == 0
    return is_valid, issues


def fix_json_file(file_path: Path) -> Tuple[bool, List[str], int]:
    """
    Attempt to fix common JSON issues in the file and rewrite it.

    Returns:
        (success, list_of_fixes_applied, questions_count)
    """
    fixes = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {str(e)}"], 0

    original_content = content

    # Try to fix trailing commas (common issue)
    # Remove trailing commas before closing brackets/braces
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    if content != original_content:
        fixes.append("Removed trailing commas")

    # Try to parse the fixed content
    try:
        questions = json.loads(content)

        # Validate and clean up the data structure
        cleaned_questions = []
        for i, item in enumerate(questions):
            if not isinstance(item, dict):
                fixes.append(f"Skipped invalid item {i+1} (not a dict)")
                continue

            # Ensure all required fields exist and are correct type
            cleaned_item = {
                "video_title": str(item.get("video_title", "")).strip(),
                "video_id": str(item.get("video_id", "")).strip(),
                "question": str(item.get("question", "")).strip(),
                "tags": [],
            }

            # Clean up tags
            tags = item.get("tags", [])
            if isinstance(tags, list):
                cleaned_item["tags"] = [
                    str(tag).strip() for tag in tags if tag and str(tag).strip()
                ]
            elif isinstance(tags, str):
                # Handle case where tags might be a string
                cleaned_item["tags"] = [tags.strip()] if tags.strip() else []
                fixes.append(f"Question {i+1}: Converted tags from string to list")

            # Only add if question is not empty
            if cleaned_item["question"]:
                cleaned_questions.append(cleaned_item)
            else:
                fixes.append(f"Question {i+1}: Removed empty question")

        # Write fixed content back
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_questions, f, indent=2, ensure_ascii=False)

        if fixes:
            fixes.append(f"Fixed file written back with {len(cleaned_questions)} questions")

        return True, fixes, len(cleaned_questions)
    except json.JSONDecodeError as e:
        return False, fixes + [f"Could not parse JSON even after fixes: {str(e)[:100]}"], 0


def main():
    """Main validation function."""
    print(f"Validating: {QUESTIONS_FILE}")
    print("-" * 60)

    is_valid, issues = validate_questions_file(QUESTIONS_FILE)

    if is_valid:
        # Count questions
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"✅ File is valid JSON!")
        print(f"   Total questions: {len(questions)}")

        # Count by video
        video_counts = Counter(q.get("video_id") for q in questions)
        print(f"   Videos: {len(video_counts)}")
        if len(video_counts) > 0:
            print(f"   Questions per video: {len(questions) / len(video_counts):.1f} average")

        # Check for any potential issues
        empty_tags = sum(1 for q in questions if not q.get("tags"))
        if empty_tags > 0:
            print(f"   ⚠️  Warning: {empty_tags} questions have no tags")

        return 0
    else:
        print(f"❌ File has {len(issues)} issue(s):\n")
        for issue in issues[:20]:  # Show first 20 issues
            print(f"   • {issue}")
        if len(issues) > 20:
            print(f"   ... and {len(issues) - 20} more issues")

        print("\n" + "-" * 60)
        print("Attempting to fix common issues...")

        success, fixes, question_count = fix_json_file(QUESTIONS_FILE)
        if success and fixes:
            print(f"✅ Applied {len(fixes)} fix(es):")
            for fix in fixes:
                print(f"   • {fix}")
            print(f"\n   Fixed file now has {question_count} questions")
            print("\nRe-running validation...")
            print("-" * 60)
            # Re-validate
            is_valid, issues = validate_questions_file(QUESTIONS_FILE)
            if is_valid:
                print("✅ File is now valid!")
                return 0
            else:
                print(f"❌ Still has {len(issues)} issue(s) after fixes")
                return 1
        else:
            print("❌ Could not automatically fix. Please fix manually.")
            if fixes:
                for fix in fixes:
                    print(f"   • {fix}")
            return 1


if __name__ == "__main__":
    sys.exit(main())


