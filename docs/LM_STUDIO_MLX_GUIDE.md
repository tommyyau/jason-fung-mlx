# Using MLX Models Directly in LM Studio

LM Studio supports MLX models directly on **Apple Silicon Macs** (M1/M2/M3), allowing you to use your fused MLX model without any conversion to GGUF or HuggingFace format.

## Prerequisites

1. **Apple Silicon Mac** (M1, M2, M3, or later)
2. **LM Studio for macOS** - Download from [lmstudio.ai](https://lmstudio.ai/)
3. **Fused MLX Model** - Your model at `models/jason_fung_mlx_fused/`

## Exact Steps

### Step 1: Verify Your Model Structure

Your fused MLX model should have these files in `models/jason_fung_mlx_fused/`:
- ✅ `config.json` - Model configuration
- ✅ `model.safetensors.index.json` - Weight index
- ✅ `model-00001-of-00002.safetensors` - Model weights (part 1)
- ✅ `model-00002-of-00002.safetensors` - Model weights (part 2)
- ✅ `tokenizer.json` - Tokenizer
- ✅ `tokenizer_config.json` - Tokenizer configuration
- ✅ `special_tokens_map.json` - Special tokens
- ✅ `chat_template.jinja` - Chat template (optional but recommended)

**Your model already has all required files!** ✅

### Step 2: Open LM Studio

1. Launch LM Studio on your Mac
2. You'll see the main interface with tabs at the top

### Step 3: Load Your MLX Model

**Method A: Using the Chat Interface (Easiest)**

1. Click on the **"Chat"** tab (or "Select a model" button)
2. Click **"Select a model"** button (usually at the top or in the left sidebar)
3. In the model selection dialog:
   - Look for a **"Browse"** or **"Open Folder"** button
   - Navigate to your project directory: `/Users/tommyyau/VSCode/jason-fung-mlx/`
   - Select the folder: `models/jason_fung_mlx_fused/`
   - Click **"Open"** or **"Select"**
4. LM Studio will load the model (this may take a moment)
5. You should see the model name appear in the model selector

**Method B: Using the Developer Tab**

1. Click on the **"Developer"** tab
2. Look for **"Load Model"** button
3. Click it and navigate to `models/jason_fung_mlx_fused/`
4. Select the directory and click **"Open"**

### Step 4: Verify Model Loaded Successfully

After loading, you should see:
- Model name displayed in the interface
- Model size information
- Status indicating the model is ready

If you see any errors, check:
- You're on an Apple Silicon Mac (required for MLX)
- LM Studio version supports MLX (v0.3.4 or later)
- All model files are present in the directory

### Step 5: Use the Model

**Option 1: Chat Interface**
- Go to the **"Chat"** tab
- Type your question/prompt
- Press Enter or click Send
- The model will respond using your fine-tuned Jason Fung style!

**Option 2: API Server (for programmatic access)**
1. Go to the **"Developer"** tab
2. Click **"Start API Server"** or **"Start Server"**
3. The server will start on `http://localhost:1234`
4. Use the OpenAI-compatible API endpoint: `http://localhost:1234/v1/chat/completions`

Example API call:
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason-fung-mlx",
    "messages": [
      {"role": "user", "content": "What is insulin resistance?"}
    ]
  }'
```

### Step 6: Configure Settings (Optional)

In the **"Developer"** tab, you can adjust:
- **Context Length** - Maximum tokens in conversation
- **Temperature** - Response randomness
- **Top P** - Nucleus sampling parameter
- **GPU Offload** - Not applicable for MLX (uses Apple Silicon automatically)

## Troubleshooting

### "Model format not supported"
- **Cause**: You're not on Apple Silicon Mac, or LM Studio version is too old
- **Solution**: Update LM Studio to v0.3.4+ and ensure you're on Apple Silicon

### "Model files not found"
- **Cause**: Incorrect path or missing files
- **Solution**: Verify all files listed in Step 1 are present in the directory

### "Failed to load model"
- **Cause**: Corrupted model files or incompatible architecture
- **Solution**: 
  1. Verify the model was fused correctly: `python3 scripts/phase4-fine-tune-model/07_fuse_lora.py`
  2. Check `config.json` is valid JSON
  3. Ensure safetensors files are not corrupted

### Model loads but gives errors
- **Cause**: Missing tokenizer files
- **Solution**: Ensure `tokenizer.json` and `tokenizer_config.json` are present

## Advantages of Using MLX Directly

✅ **No conversion needed** - Use your model immediately after fusion  
✅ **Native performance** - Optimized for Apple Silicon  
✅ **Smaller file size** - No need for GGUF conversion overhead  
✅ **Faster loading** - MLX format loads quickly on Mac  

## When to Convert to GGUF Instead

Convert to GGUF if you need:
- Cross-platform compatibility (Windows/Linux)
- Compatibility with other tools (llama.cpp, Ollama)
- Sharing with non-Mac users

For Mac-only use, MLX format is the best choice!

## Next Steps

After successfully loading your model in LM Studio:
1. Test it with questions from your training data
2. Compare responses to ground truth answers
3. Adjust temperature/sampling parameters for best results
4. Use the API server for integration with other applications





