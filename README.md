# Call Center AI Agentic Pipeline - Service-Oriented Architecture

A production-ready, on-premise service-oriented pipeline for call center audio analysis using LangGraph. This implementation uses a clean service-oriented architecture with clear separation of concerns.

## 🎯 Overview

This solution implements a complete LangGraph-based pipeline that:
- Transcribes audio using fine-tuned Whisper models 
- Refines transcripts using Gemini API (`gemini-2.0-flash-exp`) with quality scoring (temporary solution)
- Performs quality assurance with conditional loops
- Corrects low-confidence transcriptions using local LLMs (Qwen)
- Classifies call subjects using DziriBERT
- Analyzes customer satisfaction using sentiment analysis
- Ensures all processing runs locally (on-premise) for compliance

## 🏗️ Architecture

### Service-Oriented Design

The architecture is organized around **services** rather than agents/tools:

```
src/
├── core/                    # Core utilities
│   ├── config.py           # Configuration management (Pydantic)
│   ├── state.py            # State definitions
│   └── base.py             # Base service classes
│
├── services/                # Service layer
│   ├── transcription.py    # Whisper transcription service
│   ├── refinement.py       # Gemini refinement service
│   ├── correction.py       # Qwen correction service
│   ├── classification.py   # DziriBERT classification service
│   └── sentiment.py        # DziriBERT sentiment service
│
└── pipeline/                # Pipeline orchestration
    └── orchestrator.py     # LangGraph pipeline orchestrator
```

### Key Design Principles

1. **Service-Oriented**: Each functionality is a self-contained service
2. **Dependency Injection**: Services receive their configuration via constructor
3. **Base Service Pattern**: All services inherit from `BaseService`
4. **Lazy Initialization**: Models loaded on first use
5. **Clear Separation**: Core, services, and pipeline are separate concerns

## 📁 File Structure

```
.
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules (excludes venv, cache, models, data)
├── .env                     # Environment variables (optional, not tracked in git)
│
├── src/                     # Main source code
│   ├── __init__.py
│   ├── core/               # Core utilities
│   │   ├── __init__.py
│   │   ├── config.py       # Settings and configuration
│   │   ├── state.py        # State management
│   │   └── base.py         # Base service class
│   │
│   ├── services/           # Service layer
│   │   ├── __init__.py
│   │   ├── transcription.py
│   │   ├── refinement.py
│   │   ├── correction.py
│   │   ├── classification.py
│   │   └── sentiment.py
│   │
│   └── pipeline/           # Pipeline orchestration
│       ├── __init__.py
│       └── orchestrator.py
│
└── data/                    # Data directory (excluded from git)
    ├── results/             # Output directory (auto-created)
    └── logs/                 # Logs directory (auto-created)
```

**Note**: The `.gitignore` file excludes virtual environments (`cc_agentic_env/`), Python cache files (`__pycache__/`, `*.pyc`), model files, and data directories to keep the repository lightweight.

## 🚀 Usage

### Basic Usage

```bash
python main.py <audio_path>
```

### With Gemini API Key

You can provide the Gemini API key in two ways:

**Option 1: Environment Variable**
```bash
# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
python main.py ./data/audio_files/sample_call.mp3
```

**Option 2: .env File (Recommended)**
Create a `.env` file in the project root:
```bash
# .env
GEMINI_API_KEY=your-api-key-here
```

The pipeline will automatically load the API key from the `.env` file if present.

### Programmatic Usage

```python
from src.pipeline import CallAnalysisPipeline
from src.core.config import get_settings

# Get settings
settings = get_settings()

# Optionally customize settings
settings.gemini.api_key = "your-key"
settings.pipeline.confidence_threshold = 0.85

# Create and run pipeline
pipeline = CallAnalysisPipeline(settings)
result = pipeline.run("path/to/audio.mp3")
```

## ⚙️ Configuration

Configuration is managed through `src/core/config.py` using Pydantic models:

### Model Settings

- **WhisperSettings**: Transcription model configuration
  - Default model path: Local path to fine-tuned Whisper model (configured in `config.py`)
  - Supports custom model paths for on-premise deployment
- **GeminiSettings**: Gemini API configuration (temporary)
  - Model used: `gemini-2.0-flash-exp` (for transcript refinement)
  - Default in config: `gemini-2.0-flash-exp`
  - API key can be set via environment variable or `.env` file
- **LLMSettings**: Local LLM configuration (Qwen)
- **ModelSettings**: Base settings for DziriBERT models

### Pipeline Settings

- `confidence_threshold`: Minimum confidence to proceed (default: 0.90)
- `max_retry_attempts`: Maximum correction attempts (default: 2)
- `audio_sample_rate`: Target audio sample rate (default: 16000)
- `output_dir`: Results output directory
- `logs_dir`: Logs directory

### Classification Settings

Fully configurable classification schema:
- Primary categories
- Category descriptions
- Sub-categories mapping
- Default sub-categories
- Other category threshold

## 🔄 Pipeline Flow

```
START
  │
  ▼
[Transcribe] → [Refine] → [Verify]
  │                           │
  │                           ├─[confidence ≥ 0.90] → [Classify] → [Sentiment] → [Save] → END
  │                           │
  │                           ├─[confidence < 0.90 & attempts ≤ 1] → [Correct] → [Transcribe] (LOOP)
  │                           │
  │                           └─[confidence < 0.90 & attempts ≥ 2] → [Save] (MANUAL_REVIEW) → END
  │
  └─[Error] → [Save] (ERROR) → END
```

## 📊 Output Format

Results are saved as JSON files in `data/results/`:

```json
{
  "call_id": "call_audio_abc123",
  "transcript": "...",
  "refined_transcript": "...",
  "refinement_score": 0.85,
  "confidence_score": 0.95,
  "is_corrected": false,
  "subject": "COMPLAINT",
  "sub_subject": "Network Coverage",
  "satisfaction_score": 3.5,
  "status": "COMPLETE",
  "audio_path": "/path/to/audio.mp3",
  "run_count": 1
}
```

**Note**: The `refinement_score` field (0.0-1.0) indicates the quality and coherence of the refined transcript, as assessed by the Gemini refinement service.

## 🎨 Service Architecture

### BaseService

All services inherit from `BaseService` which provides:
- Automatic initialization tracking
- Error handling with timing
- Logging infrastructure
- Consistent interface

### Service Pattern

Each service:
1. Inherits from `BaseService`
2. Implements `initialize()` for model loading
3. Implements `process()` for main functionality
4. Uses `_execute_with_timing()` for automatic error handling

### Example Service

```python
class MyService(BaseService):
    def __init__(self, settings):
        super().__init__("my_service")
        self.settings = settings
    
    def initialize(self):
        # Load models, etc.
        pass
    
    def process(self, input_data):
        def _process():
            # Main processing logic
            return {"result": "..."}
        
        return self._execute_with_timing(_process)
```

## 🔧 Extending the Pipeline

### Adding a New Service

1. Create new service file in `src/services/`
2. Inherit from `BaseService`
3. Implement `initialize()` and `process()`
4. Add to `src/services/__init__.py`
5. Use in orchestrator

### Modifying Pipeline Flow

Edit `src/pipeline/orchestrator.py`:
- Add new nodes in `_build_graph()`
- Implement node methods
- Update routing logic in `_route_decision()`

## 📝 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- **LangGraph** (>=0.2.0) - Workflow orchestration
- **LangChain** (>=0.1.0) - LLM framework integration
- **Transformers** (4.57.3) - Model loading and inference
- **PyTorch** (2.9.1+cu130) - Deep learning framework with CUDA support
- **OpenAI Whisper** (20250625) - Audio transcription
- **Librosa** (0.11.0) - Audio processing
- **Pydantic** (2.12.5) - Configuration and data validation
- **Google Generative AI** - Gemini API client (for transcript refinement)
- **Hugging Face Hub** (0.36.0) - Model repository access
- **Accelerate** (1.12.0) - Model optimization
- **Datasets** (4.4.1) - Data handling utilities

**Note**: The project uses CUDA-enabled PyTorch builds. For CPU-only systems, install the CPU version of PyTorch instead.

## 🛡️ Compliance & Security

- **On-Premise Execution**: All models (except Gemini API) run locally
- **Data Residency**: Compliant with Algerian data protection laws
- **Model Caching**: Models loaded once and reused
- **Type Safety**: Pydantic models ensure data validation
- **Error Handling**: Robust error handling at all levels

## 📈 Performance Considerations

- **Lazy Loading**: Models loaded on first use
- **GPU Support**: Automatic CUDA detection
- **Model Caching**: Single model instance per service
- **Memory Management**: Appropriate precision (float16 on GPU)
- **Long Audio Handling**: Automatic chunking for audio files longer than 30 seconds

## 📄 License

This implementation is provided as-is for on-premise use in compliance with Algerian data protection laws.
