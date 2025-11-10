# The Fine-Tuning Saga: Training Challenges, Parameter Tuning, and Model Selection

## TL;DR: Key Lessons Learned

**The Journey in a Nutshell:**
This is a first-time fine-tuning story—starting from near-zero knowledge, making many mistakes, and learning through failure. Here are the critical lessons:

### 🎯 Core Principles

1. **Fine-tuning tweaks <0.1% of parameters** - It's about gentle influence, not radical change
2. **Base model capability = ceiling** - You can't make a model smarter than its base
3. **Choose the biggest capable model you can train, not the smallest** - Small models are easier to corrupt, not easier to train
4. **Catastrophic forgetting is real and common** - Even professionally-trained models on Hugging Face have this problem
5. **Training visibility is essential** - You can't train blind; configure progress reporting, validation, and checkpoints

### 🛠️ Tool Selection

6. **Sloth doesn't work on Mac/Apple Silicon** - Had to pivot to Apple MLX (released Dec 2023)
7. **MLX was a blessing** - Native Apple Silicon support, LoRA/QLoRA support, LM Studio compatible
8. **Compatibility matters as much as capability** - A model that trains well but can't be used is useless

### 📊 Parameter Tuning (All to Prevent Catastrophic Forgetting)

9. **Learning rate: 1e-5 → 5e-6** - Gentler updates preserve base knowledge
10. **LoRA layers: 16 → 12** - Freeze more bottom layers (they contain fundamental knowledge)
11. **LoRA rank: 8 → 6** - Smaller adaptation space = less aggressive changes
12. **LoRA alpha: 16 → 8** - Less aggressive scaling of adapters
13. **LoRA dropout: 0.05 → 0.1** - Better regularization prevents overfitting
14. **Gradient accumulation: Default → 8** - Smoother gradients, "flattens" learning curve
15. **Epochs: 3 → 2** - Less over-training, fewer opportunities for forgetting
16. **Batch size: 4 → 1** - Memory constraint (compensated by gradient accumulation)

### 🎓 Model Selection Journey

17. **1B Llama** - Too small, too dumb, couldn't learn
18. **Granite** - Training worked, but GGUF conversion failed (shape/size issues)
19. **Medical LLMs from Hugging Face** - Same catastrophic forgetting problems, couldn't even say "hello"
20. **Llama 3.2 3B** - Finally worked: right balance of capability, size, and compatibility

### 💡 Hard-Won Insights

21. **Starting from zero knowledge is okay** - Problems force you to learn what you need
22. **Every failure is a lesson** - Each mistake taught us about a new parameter
23. **Fine-tuning is a delicate balance** - All parameters interact; can't just use defaults
24. **Professional models can be broken too** - Just because it's on Hugging Face doesn't mean it's good
25. **Sometimes simple is best** - Proven, stable base models > fancy specialized ones

### ⚠️ Common Mistakes to Avoid

- ❌ Choosing smallest model because "it's easier" (it's not)
- ❌ Using default parameters (they're too aggressive)
- ❌ Training without visibility (you'll waste time)
- ❌ Ignoring compatibility (GGUF conversion, workflow integration)
- ❌ Thinking fine-tuning adds knowledge (it only influences style)

---

## Introduction

This document chronicles the challenges, failures, and hard-won lessons from actually training the fine-tuned model. While data preparation was complex, training the model itself presented a whole new set of problems: visibility into training progress, catastrophic forgetting, and the critical importance of choosing the right base model.

This is the story of how we went from a model that couldn't even say "hello" to a working fine-tuned model that preserves the base model's knowledge while learning the desired style.

**This is a historical record** of a first-time fine-tuning journey—all the mistakes made, all the lessons learned, and all the failures that led to eventual success. Every mistake is part of the learning process, and we're documenting them so others can learn too.

---

## Part 1: Tool Selection and Learning from Zero

### The Initial Plan: Using Sloth

**The Expectation:**
Initially, the plan was to use Sloth for fine-tuning. There were many videos and tutorials highlighting how wonderful Sloth was for fine-tuning language models. It seemed like the perfect tool for the job, with features that promised efficient fine-tuning using LoRA, QLoRA, and other modern techniques.

**The Techniques We Wanted:**
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning by training small adapter matrices
- **QLoRA**: Even more efficient variant using quantized base models
- **Parameter-efficient fine-tuning**: Techniques that modify only a small fraction of parameters

These were exactly the techniques we wanted to use for fine-tuning, as they allow training on consumer hardware while preserving the base model's knowledge.

### The Disappointment: Sloth Doesn't Work on Mac

**The Discovery:**
Sadly, after all those videos about how wonderful Sloth is, we discovered a critical problem: **Sloth doesn't work on Mac, and it doesn't work on Apple Silicon.**

This was deeply disappointing. Here we were, ready to fine-tune our model with all the right techniques, and the tool we had planned to use simply wouldn't run on our hardware.

**The Frustration:**
- Watched countless tutorials and videos
- Understood the techniques we wanted to use
- Had our data prepared and ready
- But the tool wouldn't work on our platform

### The Fortunate Discovery: Apple MLX

**The Solution:**
Fortunately, Apple had recently released a new machine learning framework called **MLX** (announced in December 2023). This turned out to be a game-changer.

**Why MLX Was Perfect:**
- **Native Apple Silicon support**: Built specifically for Apple's hardware
- **Efficient LoRA/QLoRA support**: Supports all the fine-tuning techniques we wanted
- **Optimized for Mac**: Takes full advantage of Apple Silicon's unified memory architecture
- **Compatible with LM Studio**: Works seamlessly with the tool we use to run local LLMs on MacBook

**The Timing:**
The release of MLX couldn't have come at a better time. It was exactly what we needed, and it was available when we needed it. This fortunate timing meant we didn't have to abandon our fine-tuning plans or compromise on the techniques we wanted to use.

**The Compatibility Bonus:**
MLX's compatibility with LM Studio was particularly valuable, as that's what we use to actually run local LLMs on the MacBook. This meant we could train with MLX and then use the fine-tuned model in our existing workflow without any friction.

### The Transition to MLX

**The Process:**
Once we discovered MLX, the transition was relatively straightforward:
- MLX supports LoRA and QLoRA out of the box
- The training data format was compatible
- We had our training and validation pairs ready
- The techniques we wanted to use (LoRA, QLoRA) were all supported

**What Made Training Simple:**
With MLX and our prepared training/validation pairs, the actual training setup was relatively simple. The data preparation had been the hard part—once we had clean, well-formatted Q&A pairs split into training and validation sets, training was straightforward.

**The Remaining Challenges:**
Of course, as the rest of this saga shows, we still had challenges:
- Training visibility (couldn't see progress)
- Catastrophic forgetting (model lost knowledge)
- Parameter tuning (needed gentle settings)
- Model selection (needed capable base model)

But the foundation—having MLX as a tool that actually worked on our hardware—was solid.

### The Lesson

**The Takeaway:**
Sometimes the tool you plan to use doesn't work on your platform. But sometimes, a better alternative comes along at just the right time. MLX wasn't just a workaround—it was actually a better fit for our use case, with native Apple Silicon support and seamless integration with our existing tools.

**The Silver Lining:**
While the Sloth disappointment was frustrating, discovering MLX turned out to be a blessing. It's optimized for the hardware we have, supports all the techniques we wanted, and integrates with our workflow. Sometimes the detour leads to a better destination.

### Starting from Near Zero: What I Knew (And Didn't Know)

**The Starting Point:**
I knew almost nothing about fine-tuning. The basic concept was clear: get some data, apply it to a model, and the model would be fine-tuned. But exactly what that meant in practice? I had no idea.

**What I Did Know:**
1. **LoRA (Low-Rank Adaptation)**: I was aware that LoRA meant tuning only a small proportion of the model rather than all of it, and this had really good effects. It was efficient and didn't require retraining everything.

2. **Quantization**: I knew about quantization—taking floats and converting them into integers or smaller, lower-fidelity representations of numbers. This made models faster without compromising too much on quality.

3. **Overfitting**: I had heard of overfitting and understood it was something to avoid, though I didn't fully understand how to prevent it in practice.

**What I Didn't Know:**
- How to actually configure fine-tuning parameters
- What learning rate meant or how to set it
- What batch sizes were or how they affected training
- What freezing layers meant or why you'd do it
- How to prevent catastrophic forgetting
- What gradient accumulation was
- How all these parameters interacted

### The Learning Journey Through Parameters

As we encountered problems (catastrophic forgetting, training instability), we had to learn about each parameter and what it did. Here's what we discovered:

#### Learning About Freezing Layers

**The Discovery:**
One of the most important things we learned was about **freezing layers**—keeping some layers of the model completely unchanged during training. This was a revelation.

**Why It Matters:**
- The bottom layers of a neural network contain fundamental knowledge
- The top layers handle style, formatting, and output generation
- By freezing the bottom layers, you preserve the base model's core knowledge
- Only the top layers get fine-tuned, which is where style and behavior live

**What We Did:**
We reduced the number of trainable layers from 16 to 12, effectively "freezing" more of the bottom layers. This was critical for preventing catastrophic forgetting—the bottom layers stayed exactly as they were, preserving all the base model's knowledge.

**The Insight:**
This was completely new to me. I had no idea that you could selectively train parts of a model while keeping other parts frozen. It's like teaching someone a new accent without changing their vocabulary.

#### Learning About Learning Rate

**The Discovery:**
Learning rate controls how big of steps the model takes when updating its weights. Too high, and it overshoots and corrupts existing knowledge. Too low, and it learns too slowly.

**What We Learned:**
- Learning rate determines how aggressive the updates are
- A high learning rate can cause catastrophic forgetting
- A lower learning rate means gentler, more conservative updates
- For fine-tuning, you want a low learning rate to preserve base knowledge

**What We Did:**
We reduced the learning rate from `1e-5` to `5e-6`—cutting it in half. This made the updates much gentler, preventing the model from making aggressive changes that would overwrite existing knowledge.

**The Insight:**
I learned that fine-tuning isn't about making big changes. It's about making tiny, careful adjustments. The learning rate is like the volume knob—turn it down to make gentle changes.

#### Learning About Batch Sizes and Gradient Accumulation

**The Discovery:**
This was interesting. I learned that having more batches would smooth out the learning process, but there's a memory trade-off.

**The Problem:**
- Larger batch sizes = smoother, more stable learning
- But larger batches = more memory needed
- On a 16GB MacBook Pro, we couldn't use large batches
- We had to reduce batch size from 4 to 1 to fit in memory

**The Solution: Gradient Accumulation**
This is where we learned about **gradient accumulation**—a technique that lets you simulate larger batch sizes without using more memory.

**How It Works:**
- Instead of updating after every batch, you accumulate gradients over multiple batches
- After 8 batches (with batch size 1), you update the model
- This gives you the stability of batch size 8, but only uses the memory of batch size 1
- The gradients get "flattened out" over multiple batches, making learning smoother

**What We Did:**
We set gradient accumulation to 8 steps. Combined with batch size 1, this gave us an effective batch size of 8, but used much less memory.

**The Insight:**
This was a clever workaround. I learned that you can get the benefits of larger batches (smoother learning) without the memory cost by accumulating gradients. It's like saving up small changes and applying them all at once.

#### Learning About LoRA Parameters

**LoRA Rank:**
- Controls the size of the adapter matrices
- Lower rank = smaller adaptation space = less aggressive changes
- We reduced it from 8 to 6 to make changes more conservative

**LoRA Alpha:**
- Controls how much the LoRA adapters influence the base weights
- Lower alpha = gentler influence = more preservation of base model
- We reduced it from 16 to 8 to reduce the impact of the adapters

**LoRA Dropout:**
- Regularization technique to prevent overfitting
- Higher dropout = less memorization of training data
- We increased it from 0.05 to 0.1 to better preserve general knowledge

**The Insight:**
I learned that LoRA has its own set of parameters that control how much the adapters can change the model. All of these needed to be tuned to be more conservative.

#### Learning About Epochs

**The Discovery:**
Epochs determine how many times the model sees the entire training dataset. More epochs = more learning, but also more risk of overfitting and catastrophic forgetting.

**What We Learned:**
- For fine-tuning, you don't need many epochs
- The model already knows most things—you're just teaching style
- Too many epochs can cause the model to forget base knowledge
- Less is more when it comes to fine-tuning

**What We Did:**
We reduced epochs from 3 to 2. This was enough to learn the style without risking too much forgetting.

**The Insight:**
I learned that fine-tuning isn't like training from scratch. You're making tiny adjustments, so you don't need to see the data many times. A few passes is enough.

### The Journey of Discovery

**The Process:**
Each parameter was learned through necessity:
1. Model showed catastrophic forgetting → Learned about learning rate and freezing layers
2. Training was unstable → Learned about gradient accumulation and batch sizes
3. Model was overfitting → Learned about dropout and epochs
4. Model was too aggressive → Learned about LoRA rank and alpha

**The Pattern:**
Every problem led to learning about a new parameter. Every failure taught us something new. It was a journey of discovery, not a guided tutorial.

**The Realization:**
Fine-tuning is a delicate balance. Every parameter matters, and they all interact. You can't just set defaults and hope for the best—you need to understand what each parameter does and tune them carefully.

**The Lesson:**
Starting from near-zero knowledge was actually okay. The problems we encountered forced us to learn what we needed to know. Each failure was a lesson, and each lesson made the next attempt better.

---

## Part 2: The Training Visibility Nightmare

### The Problem: Training in the Dark

**The Situation:**
Once we had our data prepared and split into training and validation sets, we thought the hard part was over. We were wrong.

Training was extremely CPU and memory intensive, making it nearly impossible to understand:
- Whether training was progressing or had crashed
- What iteration/step the training was on
- Whether the model was learning or stuck
- How much time remained
- If the process was even doing anything useful

**The Reality:**
- Training would run for hours with no clear progress indicators
- The computer would be maxed out (CPU at 100%, memory nearly full)
- No way to tell if it was working or frozen
- Had to stop the computer multiple times due to uncertainty
- Each stop created large intermediate cache files in private directories
- Eventually ran out of disk space from accumulated cache files
- No way to monitor training health or progress
- GPU timeout errors occurred (METAL command buffer execution failed)
- Training would crash with cryptic errors, making it unclear if it was a real problem or just needed to restart

**The Frustration:**
It was like trying to drive a car blindfolded. You know you're moving (the CPU is hot, memory is used), but you have no idea:
- Where you are
- How fast you're going
- If you're going in the right direction
- If you've crashed

### The Solution: Training Visibility Parameters

To address the visibility problem, we configured explicit progress reporting in the MLX training command:

#### 1. **`--steps-per-report 50`**: Report training progress every 50 steps
   - Shows loss, learning rate, and other metrics
   - Provides clear indication that training is progressing
   - Helps identify if training has stalled or crashed
   - Gives you confidence that something is happening

#### 2. **`--steps-per-eval 50`**: Run validation every 50 steps
   - Evaluates model on validation set periodically
   - Shows validation loss to monitor overfitting
   - Provides checkpoints to assess model quality during training
   - Allows you to see if the model is actually learning

#### 3. **`--save-every 500`**: Save LoRA adapters every 500 steps
   - Creates checkpoints that can be resumed from
   - Prevents loss of progress if training is interrupted
   - Allows evaluation of intermediate models
   - Means you don't lose everything if you need to stop

**The Result:**
With these visibility parameters, training became much more transparent:
- Clear progress indicators every 50 steps showing loss and metrics
- Regular validation showing model health and learning progress
- Checkpoints that could be resumed, preventing wasted time
- Ability to monitor training without stopping to check
- Confidence that training is actually working

**The Lesson:**
Training visibility is not optional—it's essential. Without it, you're flying blind and wasting time. Always configure progress reporting, validation, and checkpointing before starting a long training run.

---

## Part 3: The Catastrophic Forgetting Disaster

### The Initial Training Attempts

**The Context:**
After getting the data working, splitting it, and creating the training and validation format, we thought we were ready. The data preparation was relatively straightforward once we understood what was needed. We proceeded to train the model with default parameters.

**The Results:**
When we actually fine-tuned the model and tested it, the results were devastating:
- **Catastrophic Forgetting**: The model seemed to lose previously learned knowledge
- **Couldn't even respond to "hello"**: Complete failure of basic capabilities
- **Lost general knowledge**: Things the base model knew were gone
- **Became less capable than the base model**: Fine-tuning made it worse, not better

**The Devastation:**
This was a crushing disappointment. We had spent so much time preparing the data, and the model couldn't even respond to a simple greeting. It was clear that something was fundamentally wrong with how we were training.

### The Realization

**The Insight:**
Fine-tuning with LoRA only modifies a tiny fraction of parameters (<0.1%), but if done incorrectly, it can still corrupt the model. The key is to make **gentle, conservative adjustments** that preserve the base model's knowledge while teaching the new style.

**The Understanding:**
- Fine-tuning is NOT about adding new knowledge
- Fine-tuning is about influencing style and behavior
- You're tweaking less than 0.1% of parameters
- If you're too aggressive, you can corrupt what's already there
- The goal is gentle influence, not radical change

---

## Part 4: Parameter Modifications to Prevent Catastrophic Forgetting

After experiencing catastrophic forgetting, we made several critical parameter adjustments. Each change was aimed at making the training gentler and more conservative, preserving the base model's knowledge while teaching the new style.

### 1. Learning Rate Reduction

- **Changed from:** `1e-5` (0.00001)
- **Changed to:** `5e-6` (0.000005)
- **Reason:** Lower learning rate = gentler updates = less disruption to base model
- **Impact:** Prevents aggressive weight changes that overwrite existing knowledge
- **Philosophy:** Make smaller steps to avoid overshooting and corrupting existing weights

### 2. LoRA Layers Reduction (Freezing More Layers)

- **Changed from:** `16 layers`
- **Changed to:** `12 layers`
- **Reason:** Fewer trainable layers = more frozen layers = more preserved base model knowledge
- **Impact:** Only the top 12 layers are fine-tuned, bottom layers remain completely frozen
- **Critical insight:** This is effectively "freezing" the bottom layers to preserve core knowledge
- **Why it matters:** The bottom layers contain fundamental knowledge; the top layers handle style and output formatting

### 3. LoRA Rank Reduction

- **Changed from:** `8`
- **Changed to:** `6`
- **Reason:** Lower rank = smaller parameter space = less aggressive adaptation
- **Impact:** Reduces the capacity for the model to "forget" by limiting how much it can change
- **Trade-off:** Smaller rank means less adaptation capacity, but also less risk of corruption

### 4. LoRA Alpha Reduction

- **Changed from:** `16`
- **Changed to:** `8`
- **Reason:** Lower alpha = smaller scaling factor = gentler LoRA influence
- **Impact:** LoRA adapters have less influence, preserving more of the base model
- **Understanding:** Alpha controls how much the LoRA adapters affect the base weights

### 5. LoRA Dropout Increase

- **Changed from:** `0.05`
- **Changed to:** `0.1`
- **Reason:** Higher dropout = better regularization = prevents overfitting to training data
- **Impact:** Reduces risk of memorizing training data at the expense of general knowledge
- **Balance:** Prevents the model from becoming too specialized to the training data

### 6. Gradient Accumulation Steps Increase

- **Changed from:** Default (typically 1-4)
- **Changed to:** `8`
- **Reason:** More accumulation = more stable gradients = smoother learning
- **Impact:** Instead of updating after every batch (which can be noisy), accumulates gradients over 8 batches for more stable updates
- **Critical note:** This is the parameter that "flattens out" the learning curve - not batch size, but gradient accumulation
- **Why it helps:** Smoother gradients mean less disruptive updates, preserving more of the base model

### 7. Epochs Reduction

- **Changed from:** `3 epochs`
- **Changed to:** `2 epochs`
- **Reason:** Fewer passes over data = less opportunity for catastrophic forgetting
- **Impact:** Prevents over-training that can corrupt the base model
- **Philosophy:** Less is more - you don't need many passes to learn style

### 8. Batch Size Reduction (Memory Constraint)

- **Changed from:** `4`
- **Changed to:** `1`
- **Reason:** 16GB RAM MacBook Pro couldn't handle larger batches
- **Impact:** Combined with gradient accumulation (8 steps), effective batch size is still 8, but uses less memory
- **Note:** This was a memory constraint, not a forgetting prevention measure
- **Compensation:** Gradient accumulation makes up for the small batch size

### Summary of Parameter Changes

| Parameter | Original | Modified | Purpose |
|-----------|----------|----------|---------|
| Learning Rate | 1e-5 | **5e-6** | Gentler updates |
| LoRA Layers | 16 | **12** | Freeze more layers |
| LoRA Rank | 8 | **6** | Smaller adaptation space |
| LoRA Alpha | 16 | **8** | Less aggressive scaling |
| LoRA Dropout | 0.05 | **0.1** | Better regularization |
| Gradient Accumulation | Default | **8** | Smoother, more stable gradients |
| Epochs | 3 | **2** | Less over-training |
| Batch Size | 4 | **1** | Memory constraint (compensated by grad accumulation) |

### The Philosophy

All these changes follow the same principle: **Make tiny, gentle adjustments to preserve the base model's knowledge while teaching the new style.** 

Fine-tuning is not about adding new knowledge—it's about influencing behavior with minimal disruption. Think of it like adjusting the tone of someone's voice rather than rewriting their vocabulary.

---

## Part 5: The Critical Model Selection Lesson

### The Model Selection Journey: A Tale of Multiple Attempts

The model selection process wasn't straightforward. It involved trying multiple models, encountering various problems, and learning hard lessons along the way. Here's the full journey:

### Attempt 1: Starting with 1 Billion Parameter Llama

**The Initial Choice:**
Started with a 1 billion parameter Llama model. The logic seemed sound: small model, less memory, easier to train.

**The Reality:**
The model was **crap**. It was too small, too limited, and fundamentally incapable of learning what we needed. All the fine-tuning in the world couldn't make up for a weak foundation.

**The Lesson:**
Size matters. A 1B model just doesn't have enough capacity to be useful, even after fine-tuning.

### Attempt 2: Discovering Granite

**The Discovery:**
Found Granite models—a newly released model series that looked promising. Granite had some really cool models that seemed quite intelligent.

**The Experience:**
- Granite models were more capable than the 1B Llama
- The model seemed intelligent and promising
- Training went well with Granite

**The Problem: GGUF Conversion Nightmare**

**The Issue:**
After successfully training with Granite, we tried to convert it to GGUF format (needed for LM Studio). This is where everything fell apart.

**The Error:**
There was something wrong with the model shape or size. The conversion process failed, and we couldn't get the fine-tuned Granite model into a usable format.

**The Frustration:**
- Made it all the way through training
- Model seemed to work
- But couldn't convert it to GGUF
- Shape/size incompatibility issues
- Multiple attempts to fix it
- Nothing worked

**Technical Analysis:**
Granite uses a non-standard architecture (Multi-head Latent Attention) that creates weight shape mismatches during GGUF conversion. The llama.cpp converter expects standard Transformer format and cannot handle Granite's extra tensor dimensions. See `docs/MLX_MODEL_COMPATIBILITY.md` for detailed technical analysis.

**The Realization:**
Sometimes a model can work great for training, but if you can't use it in your workflow (LM Studio), it's useless. Compatibility matters as much as capability.

### Attempt 3: Trying Medical/Health-Focused LLMs

**The Idea:**
Since we were fine-tuning for medical education content, why not start with a model that was already trained on medical/health data? There were several available on Hugging Face:
- Doctor-focused models
- Health/medical chatbots
- Medically-trained LLMs

**The Logic:**
These models should already know medical facts, so fine-tuning them for Dr. Fung's style should be easier, right?

**The Testing:**
Downloaded several medical LLMs and tested them in LM Studio before even attempting to fine-tune.

**The Shocking Discovery:**
Oh my god. These models had the **exact same problems** we were experiencing:
- Couldn't even naturally respond to "hello"
- Showed signs of catastrophic forgetting
- Lost basic conversational abilities
- Were clearly overfitted or had catastrophic forgetting issues

**The Realization:**
This was eye-opening. These professionally-trained medical models had the same problems we were struggling with. It wasn't just us—this is a **common problem that people don't talk about**.

**The Insight:**
You can really screw up an LLM when you're giving it a lot of data to train on, or when you overfit, or when you have catastrophic forgetting. Even professionally-trained models can have these issues.

**The Disappointment:**
I had hoped that the doctor/medical chatbots available on Hugging Face would be of higher quality than what I was building. But they were just as bad—some even worse. They had clearly been fine-tuned too aggressively and lost their base capabilities.

**The Lesson:**
Just because a model is on Hugging Face doesn't mean it's good. Many fine-tuned models suffer from the same problems we were trying to solve. The fact that these professional models had the same issues validated that we weren't alone in this struggle.

### Attempt 4: Back to Basics - Llama 3.2 3B

**The Decision:**
After all these attempts, we decided to go back to basics. We chose **Llama 3.2 3B** (3 billion parameters) as the base model.

**Why This One:**
- **Proven and stable**: Llama models are well-tested and reliable
- **Right size**: 3B parameters is the practical limit for 16GB RAM
- **Good balance**: Capable enough to learn, not so large it's unusable
- **Compatible**: Works with MLX, converts to GGUF, works in LM Studio
- **Clean slate**: Starting with a good base model, not a problematic one

**The Result:**
This was the model that finally worked. It had the right balance of:
- Capability (smart enough to learn)
- Size (fits in memory)
- Compatibility (works with our tools)
- Stability (doesn't have pre-existing problems)

### The Full Journey Summary

| Attempt | Model | Result | Problem |
|---------|-------|--------|---------|
| 1 | 1B Llama | ❌ Failed | Too small, too dumb |
| 2 | Granite | ⚠️ Partial | Training worked, but GGUF conversion failed (shape/size issues) |
| 3 | Medical LLMs | ❌ Failed | Same catastrophic forgetting problems, couldn't even say "hello" |
| 4 | Llama 3.2 3B | ✅ Success | Right balance of capability, size, and compatibility |

### The Critical Lessons

**1. Base Model Quality Matters:**
- You can't fine-tune a model to be smarter than its base
- A weak base model = a weak fine-tuned model
- Even "professionally" trained models can have problems

**2. Compatibility is Critical:**
- A model that trains well but can't be used is useless
- GGUF conversion issues can kill an otherwise good model
- Make sure the model works in your workflow before committing

**3. Catastrophic Forgetting is Common:**
- It's not just beginners who have this problem
- Even professionally-trained models on Hugging Face can have catastrophic forgetting
- This is a widespread issue that people don't talk about enough

**4. Sometimes Simple is Best:**
- After trying specialized models, going back to a proven, stable base model worked
- Llama 3.2 3B wasn't fancy, but it was reliable
- Reliability > Novelty when it comes to base models

### The Core Realization

**The Evaluation Process:**
After extensive evaluation of medical facts and knowledge across all the models we tried:
- Tested base models on medical questions
- Discovered many base models got facts wrong
- Realized: **You can't fine-tune a model to be smarter than its base**
- The base model's limitations became the ceiling for the fine-tuned model

**The Hard Truth:**
All the parameter tuning, all the careful data preparation, all the time spent optimizing—none of it mattered if the base model was fundamentally limited. You can't polish a turd, and you can't fine-tune a model to be smarter than it already is.

**The Critical Lesson:**
**Choose the biggest, most capable model you can train, not the smallest.**

**Why:**
1. **Fine-tuning doesn't add knowledge** - it only tweaks <0.1% of parameters
2. **You're influencing style/behavior, not adding facts**
3. **The base model's capabilities are the ceiling** - you can't make it smarter
4. **A dumb base model = a dumb fine-tuned model**, no matter how well you tune

**The Misconception:**
Many people think "small models are easier to train" and choose the smallest model possible. This is wrong. Small models are:
- Easier to corrupt
- More limited in capability
- Harder to fine-tune successfully (less room for error)
- Not actually easier—just more limited

**The Final Choice: Llama 3.2 3B**

**Why This One Worked:**
- **Proven and stable**: Llama models are well-tested and reliable
- **Right size**: 3B parameters is the practical limit for 16GB RAM
- **Good balance**: Capable enough to learn, not so large it's unusable
- **Compatible**: Works with MLX, converts to GGUF, works in LM Studio
- **Clean slate**: Starting with a good base model, not a problematic one

**The Hardware Reality:**
- 16GB RAM MacBook Pro (5 years old)
- Training uses ~15GB of memory (pushing the limits)
- CPU temperature reaches 80+ degrees Celsius
- GPU timeout errors occurred during intensive training
- Pushing the hardware to its limits
- Would prefer 7B model, but hardware constraints limit to 3-4B
- Had to restart training multiple times due to hardware stress

**The Trade-off:**
- More memory pressure
- Hotter CPU
- Longer training times
- GPU timeouts requiring restarts
- But: **Actually capable of learning**

**The Final Insight:**
**The biggest lesson nobody tells you:** Don't choose a small model because it's "easier to train." Choose the **biggest, most capable model you can actually train** on your hardware.

Fine-tuning is about style and behavior, not adding knowledge. If your base model is dumb, your fine-tuned model will be dumb too—no amount of parameter tuning can fix a weak base model. You're tweaking <0.1% of parameters, so you need a good foundation. The dumber your base model is, the lower the ceiling of what you can actually achieve.

---

## Part 6: The Training Process - Putting It All Together

### The Final Training Configuration

After all the lessons learned, here's what the training process looked like:

**Visibility:**
- `--steps-per-report 50` - Progress every 50 steps
- `--steps-per-eval 50` - Validation every 50 steps
- `--save-every 500` - Checkpoints every 500 steps

**Gentle Parameters:**
- Learning rate: `5e-6` (reduced from 1e-5)
- LoRA layers: `12` (reduced from 16 to freeze more)
- LoRA rank: `6` (reduced from 8)
- LoRA alpha: `8` (reduced from 16)
- LoRA dropout: `0.1` (increased from 0.05)
- Gradient accumulation: `8` (increased for stability)
- Epochs: `2` (reduced from 3)

**Model Selection:**
- **Llama 3.2 3B Instruct** (not tiny LLaMA)
- Pushing hardware limits but worth it
- Capable enough to learn the desired style

### The Training Experience

**With Visibility:**
- Could see progress every 50 steps
- Knew when validation was running
- Had checkpoints to resume from
- Confidence that training was working

**With Gentle Parameters:**
- Model preserved base knowledge
- Learned the desired style
- No catastrophic forgetting
- Could actually respond to "hello" and more

**With Right Model:**
- Base model was capable
- Fine-tuning could influence style
- Results were actually useful
- Worth the hardware stress

---

## Conclusion

The fine-tuning saga taught us that:

1. **Training visibility is essential** - You can't train blind
2. **Catastrophic forgetting is real** - Gentle parameters are critical
3. **Model selection is everything** - Base model capability = ceiling
4. **Fine-tuning is gentle influence** - Not radical change
5. **Hardware constraints are real** - But push them for the best model

The journey from a model that couldn't say "hello" to a working fine-tuned model required:
- Understanding what fine-tuning actually does
- Choosing the right base model
- Configuring gentle, conservative parameters
- Having visibility into the training process
- Learning from failures and iterating

**The Final Lesson:**
Fine-tuning is not about adding knowledge. It's about gently influencing style and behavior by tweaking less than 0.1% of parameters. To do this successfully, you need:
- A capable base model (the ceiling)
- Gentle, conservative parameters (to preserve knowledge)
- Visibility into training (to know it's working)
- Patience and iteration (to get it right)

---

*This document was synthesized from the actual training experience and lessons learned during the project.*

