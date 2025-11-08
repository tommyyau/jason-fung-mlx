# The Data Refinement Journey: Lessons Learned from Fine-Tuning a Medical Education Model

## TLDR: The Journey in Bullet Points

**The Problem:** Everyone says "data refinement is important" for fine-tuning, but nobody tells you what that actually means or how hard it is.

**The Journey:**
- ❌ **Phase 1**: Tried using raw transcripts → Terrible results (filler words, errors, no structure)
- ❌ **Phase 2**: Cleaned transcripts → Still terrible (stream of words, no structure)
- ✅ **Phase 3**: Realized we need Q&A pairs → Breakthrough moment
- ❌ **Phase 4**: Tried semantic chunking → Answers split across chunks, lost context
- ❌ **Phase 5**: Tried validating medical facts → Wrong approach (fine-tuning ≠ fact addition, use RAG)
- ⚠️ **Phase 6**: Full transcript approach → Better, but single-pass extraction was messy
- 💡 **Phase 7**: Trained model, got messy unformatted responses → Realized format matters as much as content
- ✅ **Phase 8**: Two-pass solution → Extract questions first, then generate well-formatted answers

**Key Lessons:**
1. Raw transcripts don't work - need Q&A structure
2. Cleaning alone isn't enough - structure > cleanliness
3. Chunking breaks continuity - full context is crucial (common RAG→fine-tuning trap)
4. Fine-tuning teaches style/patterns, not facts - use RAG for facts
5. **Format matters as much as content - model learns formatting from training data, NOT prompts**
6. Two-pass processing works best - separate question extraction from answer generation
7. Question numbering enables batch processing with reliable synchronization
8. **Training visibility is critical - use steps-per-report, steps-per-eval, and save-every to monitor progress**
9. **Catastrophic forgetting is real - reduce learning rate, freeze layers, use gentle parameters**
10. **Choose the biggest capable model you can train, not the smallest - base model capability = ceiling**

**The Result:** A clean, efficient two-pass pipeline that produces high-quality, well-formatted Q&A pairs ready for fine-tuning.

---

## Introduction

This document chronicles the evolution of our data preparation pipeline for fine-tuning a language model on Dr. Jason Fung's medical education content. What started as a naive assumption that transcripts could be directly used for fine-tuning evolved into a sophisticated two-pass question-answer extraction system. Along the way, we learned critical lessons about what fine-tuning is actually good for, and what it's not.

## Phase 1: The Naive Beginning

### Initial Assumption: "Just Extract and Fine-Tune"

**The Plan:**
- Extract video transcripts from YouTube
- Load them directly into the fine-tuning pipeline
- Expect the model to learn Dr. Fung's teaching style and knowledge

**The Reality:**
The results were terrible. The transcripts were full of:
- Filler words: "um", "uh", "er", "you know"
- Repetitions and false starts
- Incomplete sentences
- Transcription errors
- Conversational artifacts that don't translate to written text

**The Lesson:**
Raw transcripts are not suitable for fine-tuning. They're a stream of consciousness that lacks the structure and clarity needed for a model to learn meaningful patterns.

---

## Phase 2: Cleaning and Context

### Attempt 1: Clean Up the Transcripts

**The Approach:**
- Remove filler words
- Fix obvious transcription errors
- Clean up grammar and sentence structure

**The Result:**
While cleaner, the data was still fundamentally flawed. It remained a continuous stream of words without clear structure. The fine-tuner couldn't extract meaningful patterns because:
- There was no clear question-answer structure
- Context was lost in the continuous flow
- The model had no way to understand what concepts were being taught

**The Lesson:**
Cleaning alone isn't enough. Fine-tuning requires structured data that teaches the model specific patterns. A stream of words, even clean words, doesn't provide the structure needed.

---

## Phase 3: The Q&A Realization

### The Breakthrough: Question-Answer Pairs

**The Insight:**
Fine-tuning works best with question-answer pairs. This is the standard format because:
- It teaches the model how to respond to specific queries
- It provides clear input-output relationships
- It structures the knowledge in a way the model can learn from

**The Challenge:**
Videos aren't structured as Q&A pairs. They're monologues, explanations, and teaching sessions. We needed to extract implicit questions and their corresponding answers from the content.

---

## Phase 4: The Chunking Experiment

### Attempt 2: Semantic Chunking + Q&A Extraction

**The Approach:**
1. Use semantic chunking (embedding-based similarity) to break transcripts into coherent segments
2. Extract Q&A pairs from each chunk independently
3. Combine all chunks into training data

**Why Chunking?**
The initial reasoning was influenced by several factors:

1. **Token Limit Concerns**: Worry that transcripts might exceed LLM context windows. The code even had a `split_long_text()` function with a `max_tokens: int = 6000` parameter, suggesting legitimate concern about exceeding limits.

2. **RAG Pattern Confusion**: Coming from RAG tutorials and guides, chunking is a standard pattern. It's natural to think "if chunking works for RAG, it should work for fine-tuning too." This is a common trap when transitioning from RAG to fine-tuning.

3. **Code Infrastructure**: The presence of text-splitting functions in the codebase naturally led toward chunking as a solution. When you have tools to split text, you tend to use them.

4. **Semantic Segmentation Appeal**: The idea of breaking content into semantic units (topic-focused chunks) seemed logical - each chunk would represent a distinct concept, making Q&A extraction more focused.

**The Realization:**
Chunking is fundamentally a **RAG technique, not a fine-tuning technique**. We were applying the wrong pattern to the wrong problem:
- **RAG**: Retrieve relevant chunks → answer from those chunks (chunking makes sense)
- **Fine-tuning**: Train on complete examples → model learns patterns from full context (chunking breaks continuity)

This is a common trap for people coming from RAG who try to apply RAG patterns to fine-tuning. The tools and techniques are different because the goals are different.

**The Implementation:**
- Used sentence embeddings to identify semantic boundaries
- Split text when similarity between adjacent sentences dropped below a threshold (0.78)
- Processed each chunk independently to extract Q&A pairs

**The Problem:**
This approach had a critical flaw: **answers were often split across chunks**. 

For example:
- Chunk 1 might contain: "The key question is: what controls fat storage?"
- Chunk 2 might contain: "The answer is insulin. When insulin is high..."
- Chunk 3 might contain: "...you cannot release stored energy. When insulin is low..."

The model would see incomplete answers, missing crucial context that appeared in subsequent chunks. This led to fragmented, incomplete training examples.

**The Lesson:**
Chunking breaks the continuity of explanations. When answers span multiple chunks, you lose the complete context needed for the model to learn properly.

---

## Phase 5: The Fact Validation Mistake

### Attempt 3: Validate Medical Accuracy (During Chunking Phase)

**The Context:**
While working with the chunked data approach, we were concerned about medical accuracy. We had extracted Q&A pairs from chunks, but wanted to ensure the answers were medically correct.

**The Approach:**
- Try to evaluate whether answers were "factually accurate"
- Filter out answers that might contain incorrect medical information
- Add validation steps to ensure medical accuracy
- Attempt to verify facts and filter training data based on accuracy

**The Realization:**
This was a fundamental misunderstanding of what fine-tuning does.

**The Critical Lesson:**
**Fine-tuning is NOT for adding facts. It's for teaching patterns, style, and behavior.**

- Fine-tuning tweaks a small number of parameters (via LoRA)
- It teaches the model HOW to respond, not WHAT facts to know
- All the facts are already in the pre-trained model
- Trying to add thousands of facts through fine-tuning is:
  - Inefficient (you'd need massive amounts of data)
  - Risky (can lead to catastrophic forgetting)
  - Wrong tool for the job

**The Right Tool for Facts: RAG (Retrieval-Augmented Generation)**
- RAG is designed for adding new facts and knowledge
- It uses external knowledge bases
- It doesn't modify the model weights
- It's the correct approach when you need to add specific information

**The Lesson:**
Understand what each tool is designed for:
- **Fine-tuning**: Teach style, patterns, response format, tone
- **RAG**: Add new facts, knowledge, and information

Don't use fine-tuning to add facts. Use it to teach the model how to communicate in a specific style.

---

## Phase 6: Full Transcript Approach

### Attempt 4: Process Entire Transcripts

**The Approach:**
- Treat each video transcript as a complete unit
- Use the full transcript context to extract Q&A pairs
- Let the LLM evaluate the entire transcript to determine the best questions

**The Improvement:**
This solved the chunking problem. Answers could now draw from the complete transcript, ensuring full context and complete explanations.

**The Remaining Issue:**
Trying to extract both questions AND answers in a single pass was complex and error-prone. The LLM had to:
1. Identify good questions
2. Find the corresponding answers
3. Format everything correctly
4. All in one API call

This led to:
- Inconsistent quality
- Missing questions or answers
- Difficult error handling
- Hard to debug when things went wrong

---

## Phase 7: The Formatting Realization

### The Training Disappointment

**The Context:**
After the full transcript approach, we had what seemed like good question-answer pairs. The questions were extracted from complete transcripts, the answers captured Dr. Fung's style and analogies, and the content was complete. We felt confident we had solid training data and proceeded to fine-tune the model.

**The Training Results:**
When we actually fine-tuned the model and tested it, the results were deeply disappointing:
- **Catastrophic Forgetting**: The model seemed to lose previously learned knowledge
- **Messy Responses**: Every response was a single, continuous line of text
- **Poor Readability**: Answers were hard to read and understand, even when the content was correct

**The Realization:**
This was a critical insight: **Format matters just as much as content.**

We had been so focused on:
- Getting the right questions
- Capturing the correct answers
- Preserving analogies and teaching style
- Ensuring complete context

But we completely missed that **the format of the training data directly influences the format of the model's output.**

### The Format Problem

**Video Transcripts vs Written Responses:**
- **Video transcripts** are conversational, stream-of-consciousness, continuous text
- **Written responses** need structure: paragraphs, line breaks, emphasis, lists

When we trained on answers that were formatted like transcripts (continuous text), the model learned to produce continuous text. It didn't understand that written responses need:
- Paragraph breaks (`\n\n`) to separate concepts
- **Bold text** (`**text**`) for key terms and emphasis
- Bullet points (`- `) and numbered lists (`1. `) for multiple items
- Line breaks for visual structure
- Proper formatting for readability

**The Critical Insight:**
Fine-tuning doesn't just teach content—it teaches **how to format responses**. If your training data is messy, unformatted text, your model will produce messy, unformatted text.

**The Unexpected Realization:**
We initially thought formatting would come from system prompts or user instructions. We were wrong. **The model learns formatting patterns directly from the training data itself.** 

When we added proper formatting to the training data (newlines `\n\n`, paragraphs, **bold text**, bullet points, numbered lists), the model learned to produce responses with that same formatting—even though the training data was stored as a single line with formatting markers. The model saw those formatting indicators (`\n\n`, `**text**`, `- `, `1. `) in the training examples and learned to reproduce them in its responses.

**This was never something we expected.** We thought formatting would be handled by prompts, but fine-tuning teaches the model the complete response pattern, including how to structure and format the output. The formatting must be in the training examples themselves—you can't rely on prompts to teach it.

### The Shift in Focus

**Before:** "Get the context right, get the style right, get the analogies right."

**After:** "Get the context right, get the style right, get the analogies right, AND format it properly for written responses."

This was a fundamental shift. We realized that:
1. **Conversational style** (from videos) ≠ **Written style** (for responses)
2. **Format is part of the training signal** - the model learns formatting patterns
3. **Readability matters** - well-formatted answers are more useful than perfectly accurate but poorly formatted ones
4. **Formatting needs to be explicit** in the training data

### The Path Forward

This realization led directly to the two-pass solution, where we could:
1. **Pass 1**: Focus on extracting good questions
2. **Pass 2**: Focus on generating well-formatted answers with:
   - Proper paragraph structure
   - Bold text for emphasis
   - Lists and bullet points where appropriate
   - Line breaks for readability
   - All while maintaining Dr. Fung's style and analogies

The two-pass approach allowed us to give proper attention to formatting in the answer generation step, rather than trying to do everything at once.

**The Lesson:**
Format is not an afterthought. It's a core part of what the model learns. **Training data must be formatted the way you want the model to respond, because the model learns formatting patterns from the training data itself—not from system prompts.**

This means:
- If you want paragraphs, include `\n\n` in your training data
- If you want bold text, include `**text**` markers in your training data
- If you want lists, include `- ` or `1. ` in your training data
- The model will learn these patterns and reproduce them in its responses

You can't rely on prompts to teach formatting. The formatting must be in the training examples themselves.

---

## Phase 8: The Two-Pass Solution

### The Final Approach: Separate Question Extraction and Answer Generation

**The Breakthrough:**
Split the process into two distinct, focused steps:

#### Pass 1: Extract All Questions
- Review the entire transcript
- Identify the best questions that can be answered from the content
- Generate standalone, self-contained questions
- Save questions with metadata (video_id, tags, etc.)

#### Pass 2: Generate All Answers
- Load the transcript and all questions for that video
- Pass all questions (with explicit numbering) to the LLM in a single call
- Generate all answers with proper synchronization using question numbers
- **Explicitly format answers** for written responses: paragraphs, bold text, lists, line breaks
- Match answers back to questions using the numbering system

**Why This Works:**
1. **Separation of Concerns**: Each step has a single, clear responsibility
2. **Better Quality**: Focused prompts produce better results
3. **Easier Debugging**: Can inspect questions independently of answers
4. **Efficient API Usage**: Generate all answers for a video in one call (with question numbering for sync)
5. **Complete Context**: Answers can draw from the full transcript
6. **No Broken Answers**: Since we process the full transcript, answers aren't split across chunks
7. **Proper Formatting**: Can explicitly focus on formatting in the answer generation step, ensuring readable, well-structured responses

**The Synchronization Technique:**
Using explicit question numbers (Question 1, Question 2, etc.) in both the prompt and the response ensures that:
- The LLM knows which question it's answering
- We can reliably match answers back to questions
- Multiple questions can be processed in parallel without losing track

**The Result:**
High-quality, complete Q&A pairs with full context, properly formatted for written responses (paragraphs, bold text, lists, line breaks), ready for fine-tuning.

---

## Key Lessons Learned

### 1. Raw Transcripts Don't Work
- Full of filler words, errors, and conversational artifacts
- Lack structure needed for learning
- Need significant processing

### 2. Cleaning Alone Isn't Enough
- Even clean transcripts are just streams of words
- Fine-tuning needs structured data (Q&A pairs)
- Structure is more important than cleanliness

### 3. Chunking Breaks Continuity (Common RAG Trap)
- Answers often span multiple chunks
- Losing context leads to incomplete training examples
- Full transcript context is crucial
- **Common mistake**: Applying RAG chunking patterns to fine-tuning (they're different use cases)
- Chunking works for RAG retrieval, but breaks continuity for training data

### 4. Format Matters as Much as Content (Critical Insight)
- **Training data format directly influences model output format** - the model learns formatting from examples, not prompts
- Written responses need structure: paragraphs (`\n\n`), **bold text** (`**text**`), lists (`- `, `1. `), line breaks
- Conversational transcripts ≠ formatted written responses
- **Unexpected realization**: Formatting comes from training data, not system prompts
- When you include formatting markers (newlines, bold, lists) in training data, the model learns to use them
- This was never expected—we thought prompts would handle formatting, but fine-tuning teaches complete response patterns including structure

### 5. Two-Pass Processing Works Best
- **Pass 1**: Extract all questions from full transcript
- **Pass 2**: Generate all answers with full context AND proper formatting
- Separation of concerns improves quality and debuggability

### 6. Question Numbering Enables Synchronization
- Explicit numbering in prompts and responses
- Allows processing multiple Q&A pairs in one API call
- Reliable matching of answers to questions

### 7. Fine-Tuning ≠ Fact Addition
- Fine-tuning teaches patterns, style, and behavior
- Facts are already in the pre-trained model
- Use RAG for adding new facts
- Don't try to validate/add facts during fine-tuning prep

### 8. Complete Context Matters
- Answers need full transcript context
- Incomplete answers lead to poor training data
- Full context enables complete, coherent explanations

### 9. Training Visibility is Critical
- Training is CPU/memory intensive and can appear frozen
- Use `--steps-per-report` to show progress regularly (every 50 steps)
- Use `--steps-per-eval` to validate model health periodically
- Use `--save-every` to create resumable checkpoints
- Without visibility, you can't tell if training is working or crashed

### 10. Prevent Catastrophic Forgetting with Gentle Parameters
- **Learning rate**: Reduce from 1e-5 to 5e-6 for gentler updates
- **LoRA layers**: Reduce from 16 to 12 to freeze more base layers
- **LoRA rank**: Reduce from 8 to 6 for smaller adaptation space
- **LoRA alpha**: Reduce from 16 to 8 for less aggressive scaling
- **Gradient accumulation**: Increase to 8 for smoother, more stable gradients
- **Epochs**: Reduce from 3 to 2 to prevent over-training
- **Philosophy**: Make tiny, gentle adjustments to preserve base model knowledge

### 11. Choose the Biggest Capable Model, Not the Smallest (Critical Lesson)
- **Mistake**: Starting with tiny LLaMA because "small models are easier to train"
- **Reality**: You can't fine-tune a model to be smarter than its base
- **Lesson**: Base model capability = ceiling for fine-tuned model
- **Solution**: Choose the biggest, most capable model you can actually train
- **Fine-tuning tweaks <0.1% of parameters** - it influences style, not knowledge
- **A dumb base model = a dumb fine-tuned model**, no matter how well you tune
- **Hardware constraints are real**, but push them to use the best model possible

---

## The Final Pipeline

### Phase 1: Extract Transcripts
1. Get video list from YouTube channel
2. Fetch transcripts using Supadata API
3. Store in JSONL format with metadata

### Phase 2: Refine Raw Data (Two-Pass System)
1. **Step 1**: Extract Questions
   - Process full transcript
   - Generate standalone questions
   - Save with metadata (video_id, tags)

2. **Step 2**: Generate Answers
   - Load transcript + all questions for video
   - Generate all answers in one API call
   - Use question numbering for synchronization
   - Match answers back to questions

3. **Step 3**: Format for Training
   - Convert to MLX-ready format (instruction/output format)
   - Ensure answers are properly formatted (paragraphs, bold, lists, line breaks)
   - Split into train/validation sets
   - Validate formatting consistency

### Phase 3-5: Training and Conversion
- Fine-tune model using MLX
- Convert to various formats (HuggingFace, GGUF)
- Evaluate and test

> **Note:** For detailed information about the training challenges, parameter tuning, and model selection lessons, see [The Fine-Tuning Saga](./FINE_TUNING_SAGA.md).

---

## Conclusion

The journey from naive transcript extraction to a sophisticated two-pass Q&A extraction system, and then through the challenges of actually training the model, taught us that:

1. **Structure matters more than cleanliness** - Q&A pairs are essential
2. **Context is king** - Full transcripts prevent broken answers
3. **Format matters as much as content** - Model learns formatting from training data, not system prompts (unexpected but critical insight)
4. **Separation of concerns** - Two-pass processing improves quality
5. **Right tool for the job** - Fine-tuning for style, RAG for facts
6. **Training visibility is essential** - Progress reporting prevents uncertainty and wasted time
7. **Gentle fine-tuning preserves knowledge** - Conservative parameters prevent catastrophic forgetting
8. **Model selection is critical** - Choose the biggest capable model, not the smallest (base model capability = ceiling)
9. **Iteration is learning** - Each failed attempt taught us something valuable

The final pipeline is clean, efficient, and produces high-quality training data. The training process is now visible, stable, and preserves the base model's knowledge. But more importantly, we now understand *why* each step is necessary, what fine-tuning is actually capable of, and how to do it without destroying the model's existing capabilities.

---

## Technical Details

### Why Semantic Chunking Was Initially Used

Semantic chunking was attempted for several reasons:

1. **Token Limit Concerns**: Legitimate worry that transcripts might exceed LLM context windows. The codebase even had `split_long_text()` functions with token limits (6000 tokens), which naturally led toward chunking as a solution.

2. **RAG Pattern Confusion**: Coming from RAG tutorials and documentation, chunking is a standard, well-documented pattern. It's natural to think "if chunking works for RAG retrieval, it should work for fine-tuning too." This is a **common trap** when transitioning from RAG to fine-tuning.

3. **Code Infrastructure**: Having text-splitting and semantic segmentation functions in the codebase made chunking an obvious path forward. When you have tools to split text, you tend to use them.

4. **Topic Segmentation Appeal**: The idea of breaking content into semantic units seemed logical - each chunk would represent a distinct topic, making Q&A extraction more focused.

**The Problem:**
Chunking is fundamentally a **RAG technique, not a fine-tuning technique**:
- **RAG**: Retrieve relevant chunks → answer from those chunks (chunking makes sense here)
- **Fine-tuning**: Train on complete examples → model learns patterns from full context (chunking breaks continuity)

**The Solution:**
For fine-tuning data preparation, the downsides (broken answers, lost context) outweighed the benefits. The solution was to:
- Process full transcripts (most actually fit in context windows)
- Use the two-pass approach to manage complexity without breaking continuity
- Generate all answers in one call per video (efficient and maintains full context)

**Lesson for Others:**
If you're coming from RAG and thinking about fine-tuning, remember: **RAG patterns don't always apply to fine-tuning**. Chunking is great for retrieval, but terrible for training data preparation.

### The Synchronization Pattern

The question numbering pattern ensures reliable matching:

**Input:**
```
Question 1: Why can't a calorie deficit exist?
Question 2: What controls fat storage?
Question 3: How does insulin affect weight loss?
```

**Output:**
```json
[
  {"question_number": 1, "answer": "..."},
  {"question_number": 2, "answer": "..."},
  {"question_number": 3, "answer": "..."}
]
```

This allows processing multiple Q&A pairs efficiently while maintaining reliable synchronization.

---

*This document was synthesized from the actual development journey and lessons learned during the project.*

