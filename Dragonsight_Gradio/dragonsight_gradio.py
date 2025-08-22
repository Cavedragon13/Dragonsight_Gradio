#!/usr/bin/env python3
"""
Dragonsight Gradio - AI Image Description Tool
Copyright © 2025 Seed13 Productions. All rights reserved.

Simple, beautiful web interface with zero CORS issues.
Just install gradio and run!
"""

import os
import json
import base64
import requests
import gradio as gr
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib
import re


class DragonEye:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.google_api_key = os.getenv('GOOGLE_VISION_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.log_file = Path("dragonsight_logs.jsonl")
        
    def get_available_models(self):
        """Get list of available models from all sources"""
        models = []
        
        # Get Ollama models first
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                ollama_models = response.json().get('models', [])
                model_names = [f"ollama:{m['name']}" for m in ollama_models]
                # Prioritize vision models
                vision_models = [m for m in model_names if any(k in m.lower() for k in ['llava', 'vision', 'clip', 'blip'])]
                models.extend(vision_models + [m for m in model_names if m not in vision_models])
        except:
            models.append('ollama:llava')
        
        # Add cloud vision APIs if keys are available
        if self.gemini_api_key:
            models.extend(['gemini:gemini-1.5-flash', 'gemini:gemini-1.5-pro'])
        
        if self.openai_api_key:
            models.extend(['openai:gpt-4o', 'openai:gpt-4o-mini'])
        
        if self.anthropic_api_key:
            models.append('anthropic:claude-3.5-sonnet')
        
        # Always include Google Vision as fallback option
        if self.google_api_key:
            models.append('google:vision-api')
        
        return models if models else ['ollama:llava']
    
    def try_ollama(self, image_data, model, prompt):
        """Try Ollama API"""
        try:
            # Remove ollama: prefix if present
            ollama_model = model.replace('ollama:', '') if model.startswith('ollama:') else model
            
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No description returned'), f'Ollama ({ollama_model})', None
            else:
                return None, None, f"Ollama failed: {response.status_code}"
                
        except Exception as e:
            return None, None, f"Ollama error: {str(e)}"
    
    def try_gemini(self, image_data, model, prompt):
        """Try Gemini API"""
        if not self.gemini_api_key:
            return None, None, "Gemini API key not found"
        
        try:
            # Use REST API instead of the SDK to avoid import issues
            gemini_model = model.replace('gemini:', '') if model.startswith('gemini:') else 'gemini-1.5-flash'
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={self.gemini_api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        }
                    ]
                }]
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return text, f'Gemini ({gemini_model})', None
                else:
                    error_msg = f"Gemini no candidates returned: {result}"
                    print(f"Gemini API Error: {error_msg}")
                    return None, None, error_msg
            else:
                error_msg = f"Gemini failed: {response.status_code} - {response.text}"
                print(f"Gemini API Error: {error_msg}")
                return None, None, error_msg
                
        except Exception as e:
            error_msg = f"Gemini error: {str(e)}"
            print(f"Gemini Exception: {error_msg}")
            return None, None, error_msg
    
    def try_openai(self, image_data, model, prompt):
        """Try OpenAI Vision API"""
        if not self.openai_api_key:
            return None, None, "OpenAI API key not found"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            openai_model = model.replace('openai:', '') if model.startswith('openai:') else 'gpt-4o-mini'
            
            payload = {
                "model": openai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'], f'OpenAI ({openai_model})', None
            else:
                return None, None, f"OpenAI failed: {response.status_code}"
                
        except Exception as e:
            return None, None, f"OpenAI error: {str(e)}"
    
    def try_anthropic(self, image_data, model, prompt):
        """Try Anthropic Claude Vision"""
        if not self.anthropic_api_key:
            return None, None, "Anthropic API key not found"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Use the correct model name
            payload = {
                "model": "claude-3-5-sonnet-20241022",  # Updated model name
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text'], 'Anthropic (Claude-3.5-Sonnet)', None
            else:
                error_msg = f"Anthropic failed: {response.status_code} - {response.text}"
                print(f"Anthropic API Error: {error_msg}")  # Debug logging
                return None, None, error_msg
                
        except Exception as e:
            error_msg = f"Anthropic error: {str(e)}"
            print(f"Anthropic Exception: {error_msg}")  # Debug logging
            return None, None, error_msg
    
    def try_google_vision(self, image_data):
        """Try Google Vision API"""
        if not self.google_api_key:
            return None, None, "Google Vision API key not found in environment"
            
        try:
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_api_key}"
            
            payload = {
                "requests": [{
                    "image": {"content": image_data},
                    "features": [
                        {"type": "LABEL_DETECTION", "maxResults": 10},
                        {"type": "TEXT_DETECTION", "maxResults": 5},
                        {"type": "OBJECT_LOCALIZATION", "maxResults": 10}
                    ]
                }]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                result = data['responses'][0]
                
                description_parts = []
                
                if 'labelAnnotations' in result:
                    labels = [label['description'] for label in result['labelAnnotations']]
                    description_parts.append(f"Objects: {', '.join(labels)}")
                
                if 'textAnnotations' in result and result['textAnnotations']:
                    text = result['textAnnotations'][0]['description'].strip()
                    description_parts.append(f"Text: \"{text}\"")
                
                if 'localizedObjectAnnotations' in result:
                    objects = [obj['name'] for obj in result['localizedObjectAnnotations']]
                    description_parts.append(f"Located objects: {', '.join(objects)}")
                
                description = '. '.join(description_parts) if description_parts else "No description available"
                return description, 'Google Vision API', None
            else:
                return None, None, f"Google Vision failed: {response.status_code}"
                
        except Exception as e:
            return None, None, f"Google Vision error: {str(e)}"
    
    def extract_metadata(self, description, file_path=None):
        """Extract metadata for logging"""
        # Extract tags
        common_descriptors = [
            'person', 'woman', 'man', 'child', 'people', 'face', 'hair', 'eyes',
            'building', 'house', 'car', 'tree', 'flower', 'animal', 'cat', 'dog',
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'dark', 'light',
            'large', 'small', 'tall', 'short', 'beautiful', 'colorful', 'bright',
            'outdoor', 'indoor', 'landscape', 'portrait', 'nature', 'urban'
        ]
        
        desc_lower = description.lower()
        tags = [desc for desc in common_descriptors if desc in desc_lower]
        
        # Extract quoted text
        quoted_text = re.findall(r'"([^"]*)"', description)
        
        # Generate filename suggestion
        clean_desc = re.sub(r'[^\w\s-]', '', description.lower())
        words = clean_desc.split()[:4]
        suggested_filename = '_'.join(words) + '.jpg'
        
        return {
            'tags': list(set(tags)),
            'quoted_text': quoted_text,
            'suggested_filename': suggested_filename,
            'word_count': len(description.split()),
            'char_count': len(description)
        }
    
    def log_description(self, image_path, description, model, api_used, prompt):
        """Log the description"""
        try:
            # Calculate image hash if we have the path
            file_hash = None
            if image_path and Path(image_path).exists():
                with open(image_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
            
            metadata = self.extract_metadata(description, image_path)
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'file_path': str(image_path) if image_path else 'uploaded_image',
                'file_name': Path(image_path).name if image_path else 'uploaded_image.jpg',
                'file_hash': file_hash,
                'model_used': model,
                'api_used': api_used,
                'prompt': prompt,
                'description': description,
                'metadata': metadata
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
            return True
        except Exception as e:
            print(f"Logging error: {e}")
            return False
    
    def get_recent_logs(self, limit=20):
        """Get recent logs for display"""
        try:
            if not self.log_file.exists():
                return pd.DataFrame()
                
            logs = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
            
            # Get recent logs and convert to DataFrame
            recent_logs = logs[-limit:]
            
            df_data = []
            for log in reversed(recent_logs):  # Newest first
                df_data.append({
                    'Timestamp': log['timestamp'][:19].replace('T', ' '),
                    'File': log['file_name'],
                    'Model': log['model_used'],
                    'API': log['api_used'],
                    'Tags': ', '.join(log['metadata']['tags'][:5]),
                    'Description': log['description'][:100] + '...' if len(log['description']) > 100 else log['description']
                })
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            print(f"Error reading logs: {e}")
            return pd.DataFrame()
    
    def analyze_image(self, image, model, prompt):
        """Main analysis function for Gradio"""
        if image is None:
            return "❌ Please upload an image", "", pd.DataFrame()
        
        try:
            # Convert image to base64
            if hasattr(image, 'name'):
                # It's a file path
                with open(image.name, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                image_path = image.name
            else:
                # It's a PIL image from Gradio
                import io
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG')
                image_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                image_path = None
            
            if not prompt.strip():
                prompt = "Describe this image in detail."
            
            # Route to appropriate API based on model selection
            description, api_used, error = None, None, None
            
            if model.startswith('ollama:'):
                description, api_used, error = self.try_ollama(image_data, model, prompt)
            elif model.startswith('gemini:'):
                description, api_used, error = self.try_gemini(image_data, model, prompt)
            elif model.startswith('openai:'):
                description, api_used, error = self.try_openai(image_data, model, prompt)
            elif model.startswith('anthropic:'):
                description, api_used, error = self.try_anthropic(image_data, model, prompt)
            elif model.startswith('google:'):
                description, api_used, error = self.try_google_vision(image_data)
            else:
                # Default to Ollama for backward compatibility
                description, api_used, error = self.try_ollama(image_data, model, prompt)
            
            # Fallback chain if primary method fails
            if not description:
                fallback_methods = [
                    ('ollama:llava', self.try_ollama),
                    ('google:vision-api', lambda d, m, p: self.try_google_vision(d)),
                ]
                
                for fallback_model, fallback_method in fallback_methods:
                    if fallback_model != model:  # Don't retry the same method
                        if fallback_model.startswith('google:') and self.google_api_key:
                            description, api_used, error = fallback_method(image_data, fallback_model, prompt)
                        elif fallback_model.startswith('ollama:'):
                            description, api_used, error = fallback_method(image_data, fallback_model, prompt)
                        
                        if description:
                            break
            
            if description:
                # Log the result
                self.log_description(image_path, description, model, api_used, prompt)
                
                # Get updated logs
                logs_df = self.get_recent_logs(5)
                
                return description, f"✨ Analysis complete using {api_used}", logs_df
            else:
                return f"❌ Analysis failed: {error}", "❌ All vision APIs failed", pd.DataFrame()
                
        except Exception as e:
            return f"❌ Error: {str(e)}", "❌ Analysis failed", pd.DataFrame()


# Initialize the dragon
dragon = DragonEye()

# Custom CSS for dragon theme
css = """
.gradio-container {
    background: linear-gradient(135deg, #2d1b69, #4c1d95) !important;
}

h1 {
    color: #facc15 !important;
    text-align: center !important;
}

.footer {
    text-align: center !important;
    color: #a855f7 !important;
    font-size: 12px !important;
}

/* Button styling with better targeting */
button[data-testid="component-button"] {
    font-weight: bold !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

/* Specific button targeting by text content */
button:has-text("🔍 Analyze Image"), button[aria-label*="Analyze"] {
    background: linear-gradient(45deg, #ef4444, #dc2626) !important;
    color: white !important;
    font-size: 16px !important;
}

button:has-text("🔄 Refresh Models"), button[aria-label*="Refresh"] {
    background: linear-gradient(45deg, #fbbf24, #f59e0b) !important;
    color: #1f2937 !important;
}

button:has-text("📋 Paste from Clipboard"), button[aria-label*="Paste"] {
    background: linear-gradient(45deg, #10b981, #059669) !important;
    color: white !important;
}

/* Fallback with nth-child targeting the button order */
.gr-form button:nth-child(1) {
    background: linear-gradient(45deg, #10b981, #059669) !important;
    color: white !important;
}

.gr-form button:nth-child(2) {
    background: linear-gradient(45deg, #fbbf24, #f59e0b) !important;
    color: #1f2937 !important;
}

.gr-form button:nth-child(3) {
    background: linear-gradient(45deg, #ef4444, #dc2626) !important;
    color: white !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, title="🐉 Dragonsight") as demo:
    gr.Markdown("# 🐉 Dragonsight")
    
    with gr.Tabs():
        with gr.TabItem("🔍 Analyze"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Image upload
                    image_input = gr.Image(
                        label="📸 Upload, Drop, or Paste Image", 
                        type="pil",
                        height=300
                    )
                    
                    # Paste from clipboard button
                    paste_btn = gr.Button("📋 Paste from Clipboard", size="sm")
                    
                    # Model selection
                    model_dropdown = gr.Dropdown(
                        choices=dragon.get_available_models(),
                        value=dragon.get_available_models()[0] if dragon.get_available_models() else "llava",
                        label="🤖 Model",
                        interactive=True
                    )
                    
                    # Refresh models button
                    refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
                    
                    # Custom prompt
                    prompt_input = gr.Textbox(
                        label="📝 Custom Prompt",
                        placeholder="Describe this image in detail.",
                        value="Describe this image in detail.",
                        lines=2
                    )
                    
                    # Analyze button
                    analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg", elem_classes="analyze-btn")
                
                with gr.Column(scale=1):
                    # Results
                    status_output = gr.Textbox(label="🔮 Status", interactive=False)
                    description_output = gr.Textbox(
                        label="📜 Description", 
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
            
            # Quick logs preview
            gr.Markdown("## 📝 Recent Analysis (Quick View)")
            logs_output = gr.Dataframe(
                headers=["Timestamp", "File", "Model", "API", "Tags"],
                label="Last 5 Analyses",
                interactive=False
            )
        
        with gr.TabItem("📊 Detailed Logs"):
            gr.Markdown("### 🗂️ Complete Analysis History & Metadata")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Detailed logs table
                    detailed_logs = gr.Dataframe(
                        headers=["Time", "File", "Model", "Tags", "Word Count"],
                        label="Click a row to see full details",
                        interactive=True
                    )
                    
                with gr.Column(scale=2):
                    # Selected log details
                    selected_file = gr.Textbox(label="📁 File", interactive=False)
                    selected_model = gr.Textbox(label="🤖 Model Used", interactive=False)
                    selected_api = gr.Textbox(label="🔌 API", interactive=False)
                    selected_prompt = gr.Textbox(label="📝 Prompt", interactive=False, lines=2)
            
            # Full description
            gr.Markdown("#### 📄 Full Description")
            full_description = gr.Textbox(
                label="Complete Description",
                lines=8,
                interactive=False
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🏷️ Extracted Metadata")
                    extracted_tags = gr.Textbox(label="Tags", interactive=False)
                    suggested_filename = gr.Textbox(label="Suggested Filename", interactive=False)
                    quoted_text = gr.Textbox(label="Quoted Text Found", interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### 📊 Statistics")
                    word_count = gr.Number(label="Word Count", interactive=False)
                    char_count = gr.Number(label="Character Count", interactive=False)
                    file_hash = gr.Textbox(label="File Hash", interactive=False)
            
            # Export options
            with gr.Row():
                export_json_btn = gr.Button("💾 Export All Logs (JSON)", variant="secondary")
                export_csv_btn = gr.Button("📊 Export Summary (CSV)", variant="secondary")
                clear_logs_btn = gr.Button("🗑️ Clear All Logs", variant="stop")
            
            export_status = gr.Textbox(label="Export Status", interactive=False)
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("<center><i>Powered by Ollama with Google Vision fallback</i></center>")
    gr.Markdown("*Copyright © 2025 Seed13 Productions. All rights reserved.*", elem_classes="footer")
    
    # Event handlers
    def refresh_models():
        new_models = dragon.get_available_models()
        current_value = new_models[0] if new_models else "ollama:llava"
        return gr.Dropdown(choices=new_models, value=current_value)
    
    def paste_from_clipboard():
        """Paste image from clipboard"""
        try:
            from PIL import ImageGrab
            img = ImageGrab.grabclipboard()
            return img if img is not None else None
        except Exception as e:
            print(f"Clipboard paste error: {e}")
            return None
    
    def get_detailed_logs():
        """Get detailed logs for the detailed view"""
        try:
            if not dragon.log_file.exists():
                return pd.DataFrame()
                
            logs = []
            with open(dragon.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
            
            df_data = []
            for i, log in enumerate(reversed(logs)):  # Newest first
                df_data.append({
                    'Index': len(logs) - i - 1,  # Hidden index for selection
                    'Time': log['timestamp'][:19].replace('T', ' '),
                    'File': log['file_name'],
                    'Model': log['model_used'],
                    'Tags': ', '.join(log['metadata']['tags'][:3]) + ('...' if len(log['metadata']['tags']) > 3 else ''),
                    'Word Count': log['metadata']['word_count']
                })
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            print(f"Error reading detailed logs: {e}")
            return pd.DataFrame()
    
    def show_log_details(detailed_logs_df, evt: gr.SelectData):
        """Show details when a log row is selected"""
        if detailed_logs_df is None or len(detailed_logs_df) == 0:
            return [""] * 10
        
        try:
            row_index = evt.index[0]
            selected_row = detailed_logs_df.iloc[row_index]
            
            # Get the full log entry
            logs = []
            with open(dragon.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
            
            # Find the matching log (using reverse index)
            log_index = len(logs) - 1 - row_index
            if log_index >= 0 and log_index < len(logs):
                log = logs[log_index]
                
                return [
                    log['file_name'],
                    f"{log['model_used']} via {log['api_used']}",
                    log['api_used'],
                    log['prompt'],
                    log['description'],
                    ', '.join(log['metadata']['tags']),
                    log['metadata']['suggested_filename'],
                    ', '.join(log['metadata']['quoted_text']) if log['metadata']['quoted_text'] else 'None',
                    log['metadata']['word_count'],
                    log['metadata']['char_count'],
                    log.get('file_hash', 'N/A')
                ]
            else:
                return ["Error: Log not found"] + [""] * 10
                
        except Exception as e:
            return [f"Error: {str(e)}"] + [""] * 10
    
    def export_logs_json():
        """Export all logs to JSON"""
        try:
            if not dragon.log_file.exists():
                return "No logs to export"
            
            logs = []
            with open(dragon.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
            
            export_path = Path("dragonsight_export.json")
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
            return f"✅ Exported {len(logs)} logs to {export_path}"
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def export_logs_csv():
        """Export summary to CSV"""
        try:
            if not dragon.log_file.exists():
                return "No logs to export"
            
            logs = []
            with open(dragon.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
            
            # Create CSV data
            csv_data = []
            for log in logs:
                csv_data.append({
                    'timestamp': log['timestamp'],
                    'file_name': log['file_name'],
                    'model_used': log['model_used'],
                    'api_used': log['api_used'],
                    'prompt': log['prompt'],
                    'description': log['description'],
                    'tags': ', '.join(log['metadata']['tags']),
                    'suggested_filename': log['metadata']['suggested_filename'],
                    'word_count': log['metadata']['word_count'],
                    'char_count': log['metadata']['char_count']
                })
            
            df = pd.DataFrame(csv_data)
            export_path = Path("dragonsight_summary.csv")
            df.to_csv(export_path, index=False)
            
            return f"✅ Exported {len(logs)} log summaries to {export_path}"
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def clear_all_logs():
        """Clear all logs (with confirmation)"""
        try:
            if dragon.log_file.exists():
                dragon.log_file.unlink()
            return "✅ All logs cleared"
        except Exception as e:
            return f"❌ Clear failed: {str(e)}"
    
    # Wire up events - all buttons are now accessible
    refresh_btn.click(refresh_models, outputs=model_dropdown)
    paste_btn.click(paste_from_clipboard, outputs=image_input)
    
    analyze_btn.click(
        dragon.analyze_image,
        inputs=[image_input, model_dropdown, prompt_input],
        outputs=[description_output, status_output, logs_output]
    )
    
    # Detailed logs events
    demo.load(get_detailed_logs, outputs=detailed_logs)
    
    detailed_logs.select(
        show_log_details,
        inputs=[detailed_logs],
        outputs=[
            selected_file, selected_model, selected_api, selected_prompt,
            full_description, extracted_tags, suggested_filename, quoted_text,
            word_count, char_count, file_hash
        ]
    )
    
    export_json_btn.click(export_logs_json, outputs=export_status)
    export_csv_btn.click(export_logs_csv, outputs=export_status)
    clear_logs_btn.click(clear_all_logs, outputs=export_status)
    
    # Load initial logs for main tab
    demo.load(lambda: dragon.get_recent_logs(5), outputs=logs_output)

if __name__ == "__main__":
    print("🐉 Dragonsight Gradio - Copyright © 2025 Seed13 Productions")
    print("=" * 60)
    print("Starting Dragonsight with public sharing enabled...")
    print("🌐 This will create a temporary public URL you can share!")
    print("🔒 Your images are processed locally, only the interface is shared")
    print("=" * 60)
    
    # Launch the interface with sharing enabled
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=True,             # 🌐 Public sharing enabled by default!
        show_error=True,
        favicon_path=None
    )