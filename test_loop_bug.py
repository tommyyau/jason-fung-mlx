#!/usr/bin/env python3
"""
Test if the training loop has a bug with step limits
"""

print("="*70)
print("TESTING TRAINING LOOP LOGIC")
print("="*70)

# Simulate the current code
print("\nCURRENT CODE (potentially buggy):")
print("-" * 70)

epochs = 2
steps = 150
data_size = 300
grad_accum = 4

current_step = 0
examples_processed = 0

for epoch in range(epochs):
    print(f"\nStarting epoch {epoch + 1}")
    for i in range(data_size):
        if current_step >= steps:
            print(f"  Breaking at example {i + 1}, current_step={current_step}")
            break

        examples_processed += 1

        # Simulate gradient accumulation
        if (examples_processed % grad_accum) == 0:
            current_step += 1

    if current_step >= steps:
        print(f"Epoch {epoch + 1} completed with current_step={current_step}")
    else:
        print(f"Epoch {epoch + 1} completed normally")

print(f"\n Final: current_step={current_step}, examples_processed={examples_processed}")
print(f" Expected: steps=150, examples=600 (2 epochs × 300)")

if examples_processed == 600:
    print("❌ BUG: Processed full 2 epochs even though steps=150!")
    print("   Should have stopped at 150 steps = 600 examples")
elif examples_processed == 600:
    print("✅ Correct: Stopped at exactly 150 steps")
else:
    print(f"⚠️  Unexpected: Processed {examples_processed} examples")

# What should happen
print("\n" + "="*70)
print("CORRECT BEHAVIOR:")
print("-" * 70)

current_step = 0
examples_processed = 0
should_stop = False

for epoch in range(epochs):
    if should_stop:
        break
    print(f"\nStarting epoch {epoch + 1}")
    for i in range(data_size):
        if current_step >= steps:
            print(f"  Breaking at example {i + 1}, current_step={current_step}")
            should_stop = True
            break

        examples_processed += 1

        # Simulate gradient accumulation
        if (examples_processed % grad_accum) == 0:
            current_step += 1

print(f"\nFinal: current_step={current_step}, examples_processed={examples_processed}")

if examples_processed == 600:
    print("✅ Correct: Stopped at exactly 600 examples (150 steps)")
else:
    print(f"❌ Wrong: Should be 600, got {examples_processed}")

print("="*70)
