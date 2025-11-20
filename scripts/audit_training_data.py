#!/usr/bin/env python3
"""
Audit training data to identify low-value examples that could be replaced
with insulin-focused comparative examples.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# Define keyword categories
CATEGORIES = {
    'insulin_vs_cico': {
        'keywords': ['insulin model', 'hormonal theory', 'cico', 'calories in.*calories out',
                    'calorie counting fails', 'calories.*incomplete', 'why.*calorie.*fail'],
        'value': 'HIGH - Directly compares insulin vs CICO',
        'keep': True
    },
    'insulin_mechanism': {
        'keywords': ['insulin', 'hormone', 'thermostat', 'set weight', 'set point',
                    'homeostasis', 'body fat.*regulat'],
        'value': 'HIGH - Teaches insulin model',
        'keep': True
    },
    'fasting_practice': {
        'keywords': ['fast', 'intermittent', 'eating window', 'time.*restrict'],
        'value': 'MEDIUM - Application of insulin model',
        'keep': True
    },
    'low_carb': {
        'keywords': ['low.*carb', 'keto', 'carbohydrate.*restrict', 'atkins'],
        'value': 'MEDIUM - Application of insulin model',
        'keep': True
    },
    'related_health': {
        'keywords': ['diabetes', 'metabolic', 'obesity', 'weight.*loss', 'blood.*sugar'],
        'value': 'MEDIUM - Related to core topics',
        'keep': True
    },
    'nutrition_general': {
        'keywords': ['protein', 'fat', 'fiber', 'nutrient', 'vitamin', 'mineral',
                    'food.*quality', 'ultra.*process', 'whole.*food'],
        'value': 'LOW-MEDIUM - General nutrition',
        'keep': 'depends'
    },
    'peripheral_health': {
        'keywords': ['sleep', 'stress', 'exercise', 'circadian', 'inflammation'],
        'value': 'LOW - Peripheral topics',
        'keep': 'some'
    },
    'very_specific': {
        'keywords': ['magnesium', 'water.*soften', 'psyllium', 'green.*tea',
                    'fasting.*mimick.*box', 'LDL.*particle'],
        'value': 'VERY LOW - Too specific/tangential',
        'keep': False
    }
}

def categorize_example(question, answer):
    """Categorize an example by its content."""
    combined_text = (question + ' ' + answer).lower()

    categories_found = []
    for cat_name, cat_info in CATEGORIES.items():
        for keyword in cat_info['keywords']:
            if keyword in combined_text or any(kw in combined_text for kw in keyword.split('.*')):
                categories_found.append(cat_name)
                break

    if not categories_found:
        return 'uncategorized'

    # Return highest priority category
    priority_order = ['insulin_vs_cico', 'insulin_mechanism', 'fasting_practice',
                     'low_carb', 'related_health', 'nutrition_general',
                     'peripheral_health', 'very_specific']

    for cat in priority_order:
        if cat in categories_found:
            return cat

    return categories_found[0]

def analyze_training_data(jsonl_path):
    """Analyze training data and categorize examples."""

    examples_by_category = defaultdict(list)

    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f, 1):
            data = json.loads(line)

            # Handle different formats
            if 'messages' in data:
                # Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                messages = data['messages']
                question = messages[0]['content'] if messages[0]['role'] == 'user' else messages[1]['content']
                answer = messages[1]['content'] if messages[1]['role'] == 'assistant' else messages[0]['content']

            elif 'text' in data:
                # Format: {"text": "<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>"}
                text = data['text']

                # Extract user and model turns
                user_match = re.search(r'<start_of_turn>user\n(.*?)<end_of_turn>', text, re.DOTALL)
                model_match = re.search(r'<start_of_turn>model\n(.*?)<end_of_turn>', text, re.DOTALL)

                question = user_match.group(1).strip() if user_match else ""
                answer = model_match.group(1).strip() if model_match else ""
            else:
                print(f"⚠️ Warning: Unknown format at line {idx}")
                continue

            category = categorize_example(question, answer)

            examples_by_category[category].append({
                'line_number': idx,
                'question': question,
                'answer': answer,
                'question_length': len(question),
                'answer_length': len(answer)
            })

    return examples_by_category

def print_statistics(examples_by_category, total_examples):
    """Print statistics about the dataset."""

    print("\n" + "="*80)
    print("TRAINING DATA AUDIT - CATEGORY BREAKDOWN")
    print("="*80 + "\n")

    # Sort by count
    sorted_cats = sorted(examples_by_category.items(), key=lambda x: len(x[1]), reverse=True)

    for cat_name, examples in sorted_cats:
        count = len(examples)
        percentage = (count / total_examples) * 100

        cat_info = CATEGORIES.get(cat_name, {'value': 'UNKNOWN', 'keep': '?'})
        value = cat_info['value']
        keep = cat_info['keep']

        print(f"{cat_name.upper()}")
        print(f"  Count: {count} ({percentage:.1f}%)")
        print(f"  Value: {value}")
        print(f"  Recommendation: {'✅ Keep all' if keep == True else '⚠️ Keep some' if keep == 'depends' or keep == 'some' else '❌ Replace most'}")
        print()

def find_low_value_examples(examples_by_category, max_to_show=10):
    """Identify specific low-value examples."""

    print("\n" + "="*80)
    print("LOW-VALUE EXAMPLES TO CONSIDER REPLACING")
    print("="*80 + "\n")

    low_value_categories = ['very_specific', 'peripheral_health', 'nutrition_general', 'uncategorized']

    replacement_candidates = []

    for cat_name in low_value_categories:
        examples = examples_by_category.get(cat_name, [])
        if examples:
            print(f"\n--- {cat_name.upper()} ({len(examples)} examples) ---\n")

            for example in examples[:max_to_show]:
                print(f"Line {example['line_number']}:")
                print(f"  Q: {example['question'][:120]}...")
                print(f"  A: {example['answer'][:120]}...")
                print()

                replacement_candidates.append(example['line_number'])

    return replacement_candidates

def analyze_insulin_vs_cico_coverage(examples_by_category):
    """Check how many examples explicitly compare insulin to CICO."""

    print("\n" + "="*80)
    print("INSULIN VS CICO COMPARATIVE EXAMPLES")
    print("="*80 + "\n")

    comparative_examples = examples_by_category.get('insulin_vs_cico', [])

    print(f"Current count: {len(comparative_examples)}")
    print(f"Recommended: 200-300 examples")
    print(f"GAP: {200 - len(comparative_examples)} examples needed\n")

    if comparative_examples:
        print("Examples found:")
        for example in comparative_examples[:5]:
            print(f"  Line {example['line_number']}: {example['question'][:80]}...")
    else:
        print("⚠️ NO explicit insulin vs CICO comparative examples found!")
        print("   This is likely why your model defaults to CICO on indirect questions.")

def generate_summary(examples_by_category, total_examples):
    """Generate actionable summary."""

    print("\n" + "="*80)
    print("ACTIONABLE SUMMARY")
    print("="*80 + "\n")

    # Count high, medium, low value examples
    high_value = sum(len(examples_by_category.get(cat, []))
                     for cat in ['insulin_vs_cico', 'insulin_mechanism'])

    medium_value = sum(len(examples_by_category.get(cat, []))
                       for cat in ['fasting_practice', 'low_carb', 'related_health'])

    low_value = sum(len(examples_by_category.get(cat, []))
                    for cat in ['nutrition_general', 'peripheral_health', 'very_specific', 'uncategorized'])

    print(f"Total examples: {total_examples}\n")

    print(f"HIGH value (keep all):     {high_value:4d} ({high_value/total_examples*100:5.1f}%)")
    print(f"MEDIUM value (keep):       {medium_value:4d} ({medium_value/total_examples*100:5.1f}%)")
    print(f"LOW value (replace some):  {low_value:4d} ({low_value/total_examples*100:5.1f}%)\n")

    # Recommendations
    comparative_count = len(examples_by_category.get('insulin_vs_cico', []))
    comparative_needed = max(0, 200 - comparative_count)

    replaceable = min(low_value, comparative_needed + 100)  # Extra buffer for diversity

    print("RECOMMENDATIONS:")
    print(f"  1. Keep {high_value + medium_value} high/medium value examples ✅")
    print(f"  2. Replace ~{replaceable} low-value examples with insulin-focused content 🔄")
    print(f"  3. Add ~{comparative_needed} explicit 'insulin vs CICO' comparisons 🎯")
    print(f"  4. Keep ~{low_value - replaceable} diverse examples to prevent catastrophic forgetting ⚖️")

    print(f"\nFinal target distribution:")
    print(f"  - High value (insulin core): ~60-70%")
    print(f"  - Medium value (applications): ~20-30%")
    print(f"  - Low value (diversity): ~10-20%")

def main():
    # Path to training data
    train_path = Path(__file__).parent.parent / 'data' / 'mlx_training_data' / 'train.jsonl'

    if not train_path.exists():
        print(f"❌ Training data not found at: {train_path}")
        return

    print(f"Analyzing: {train_path}")

    # Analyze
    examples_by_category = analyze_training_data(train_path)
    total_examples = sum(len(examples) for examples in examples_by_category.values())

    # Print results
    print_statistics(examples_by_category, total_examples)
    analyze_insulin_vs_cico_coverage(examples_by_category)
    replacement_candidates = find_low_value_examples(examples_by_category, max_to_show=15)
    generate_summary(examples_by_category, total_examples)

    # Save detailed results
    output_path = Path(__file__).parent.parent / 'training_data_audit.json'

    audit_results = {
        'total_examples': total_examples,
        'category_counts': {cat: len(examples) for cat, examples in examples_by_category.items()},
        'replacement_candidates': replacement_candidates,
        'examples_by_category': {
            cat: [{'line': ex['line_number'], 'question': ex['question'][:200]}
                  for ex in examples[:50]]  # First 50 of each category
            for cat, examples in examples_by_category.items()
        }
    }

    with open(output_path, 'w') as f:
        json.dump(audit_results, f, indent=2)

    print(f"\n📊 Detailed audit saved to: {output_path}")

if __name__ == '__main__':
    main()
