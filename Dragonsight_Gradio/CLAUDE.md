# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Dragonsight_Gradio is an AI-powered image analysis tool that provides a beautiful web interface for describing images using multiple AI vision models. It supports local Ollama models, cloud APIs (Gemini, OpenAI, Anthropic), and Google Vision API with automatic fallback between services.

## Commands

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv dragonsight_env
source dragonsight_env/bin/activate  # On Windows: dragonsight_env\Scripts\activate

# Install dependencies
pip install gradio pandas requests pillow

# Optional: Install clipboard support for image pasting
pip install pillow  # PIL ImageGrab for clipboard functionality
```

### Local Development

```bash
# Run the application locally
python dragonsight_gradio.py

# Run with specific port
python dragonsight_gradio.py --server-port 7861

# Access the interface
open http://localhost:7860
```

### API Key Configuration

```bash
# Set environment variables for cloud APIs (optional)
export GOOGLE_VISION_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here" 
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"

# For local Ollama (default backend)
# Ensure Ollama is running on localhost:11434
ollama serve
ollama pull llava  # Install a vision model
```

### Testing & Validation

```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# Test with a sample image
python -c "
import requests
import base64
with open('test_image.jpg', 'rb') as f:
    data = base64.b64encode(f.read()).decode()
response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'llava',
    'prompt': 'Describe this image',
    'images': [data],
    'stream': False
})
print(response.json())
"
```

### Deployment

```bash
# Deploy to Hugging Face Spaces (requires HF_TOKEN)
gradio deploy

# Manual deployment setup
export HF_TOKEN="your_huggingface_token"
gradio deploy --space-id your-username/dragonsight
```

### Log Management

```bash
# View recent logs
tail -f dragonsight_logs.jsonl

# Export logs
python -c "
from dragonsight_gradio import DragonEye
dragon = DragonEye()
logs_df = dragon.get_recent_logs(50)
print(logs_df)
"

# Clear logs
rm dragonsight_logs.jsonl
```

## Architecture

### Core Components

**DragonEye Class**: Main analysis engine that orchestrates multiple AI vision APIs
- **Model Detection**: Automatically discovers available Ollama models and cloud APIs
- **API Routing**: Routes requests to appropriate vision service based on model selection
- **Fallback Chain**: Automatically tries alternative services if primary fails
- **Logging System**: Comprehensive logging with metadata extraction and analysis history

**Gradio Interface**: Two-tab web UI built with Gradio blocks
- **Analysis Tab**: Main interface for image upload, model selection, and description generation
- **Logs Tab**: Detailed history view with metadata, export capabilities, and log management

### AI Vision Backends

The application supports multiple vision APIs with automatic fallback:

1. **Ollama (Primary)**: Local models via HTTP API at localhost:11434
   - Prioritizes vision-capable models (llava, vision, clip, blip)
   - Supports any Ollama model with vision capabilities
   
2. **Cloud APIs (Optional)**: Requires API keys in environment variables
   - **Gemini**: Google's Gemini 1.5 Flash/Pro models via REST API
   - **OpenAI**: GPT-4o and GPT-4o-mini via Chat Completions API
   - **Anthropic**: Claude 3.5 Sonnet via Messages API
   - **Google Vision**: Classic Vision API for object/text detection

### Data Flow

```
Image Upload → Base64 Encoding → Model Selection → API Call → Description → Metadata Extraction → Logging → Display
```

### Logging & Metadata System

**JSONL Log Format**: Each analysis creates a structured log entry in `dragonsight_logs.jsonl`
- Timestamp, file info, model used, API endpoint
- Full prompt and generated description
- Extracted metadata (tags, quoted text, suggested filename)
- File hash for duplicate detection
- Word/character counts for analysis statistics

**Metadata Extraction**: Automatic extraction from descriptions
- Common descriptors (colors, objects, people, settings)
- Quoted text detection for OCR-like functionality
- Filename suggestions based on description content
- Statistical analysis (word count, character count)

### UI Architecture

**CSS Theming**: Dragon-themed gradient design with purple/gold color scheme
- Custom button styling with gradients
- Responsive layout with two-column design
- Dark theme optimized for long usage sessions

**Component Organization**:
- Image upload with drag-drop and clipboard paste support
- Dynamic model dropdown with refresh capability
- Customizable prompt input with default template
- Real-time status updates and result display
- Tabbed interface for analysis vs. logs

### Error Handling & Resilience

**Graceful Degradation**: Multiple fallback layers ensure analysis always works
- Primary model fails → Try fallback models
- All Ollama models fail → Try cloud APIs
- All APIs fail → Return informative error message

**Timeout Management**: All API calls have appropriate timeouts
- Ollama: 60 seconds (local processing can be slow)
- Cloud APIs: 30 seconds (network dependent)

**Image Format Support**: Handles multiple input methods
- File upload via Gradio interface
- Drag and drop functionality
- Clipboard paste (requires PIL ImageGrab)
- Automatic conversion to JPEG format for API compatibility

## Configuration & Customization

### Model Priority Configuration

The `get_available_models()` method automatically prioritizes vision models. To customize model selection:

```python
# Modify model detection in DragonEye.__init__()
# Add custom model preferences or API endpoints
```

### Prompt Templates

Default prompt: "Describe this image in detail."
Customize in the Gradio interface or modify the default in `analyze_image()` method.

### Logging Configuration

Log file location and format can be customized in `DragonEye.__init__()`:
```python
self.log_file = Path("custom_logs.jsonl")  # Change log file name
```

### UI Customization

Modify the CSS section (lines 459-514) to change:
- Color scheme and gradients
- Button styling and hover effects  
- Layout spacing and typography
- Theme colors and background patterns

## Security Considerations

**API Key Management**: All API keys are read from environment variables, never hardcoded
**Local Processing**: Image analysis happens locally when using Ollama
**Public Sharing**: Default configuration enables Gradio's public sharing feature
**File Handling**: Uploaded images are processed in memory, not saved to disk permanently

## Troubleshooting

### Common Issues

**Ollama Connection Failed**: 
```bash
# Ensure Ollama is running
ollama serve
# Check if models are installed
ollama list
```

**No Vision Models Available**:
```bash
# Install a vision-capable model
ollama pull llava
ollama pull moondream
```

**Cloud API Failures**: Check API key configuration and account limits
```bash
echo $GEMINI_API_KEY  # Verify environment variables are set
```

**Clipboard Paste Not Working**: Install PIL with ImageGrab support
```bash
pip install pillow[clipboard]
```

### Performance Optimization

**Local Processing**: Use Ollama with GPU acceleration for faster analysis
**Model Selection**: Smaller models (moondream) process faster than larger ones (llava:34b)
**Batch Processing**: Process multiple images by repeatedly using the same interface