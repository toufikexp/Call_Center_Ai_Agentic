# Call Center AI Agentic Pipeline - Service-Oriented Architecture

A production-ready, on-premise service-oriented pipeline for call center audio analysis using LangGraph. This implementation uses a clean service-oriented architecture with clear separation of concerns.

## 🎯 Overview

This solution implements a complete LangGraph-based pipeline that:
- Transcribes audio using fine-tuned Whisper models 
- Refines transcripts using Gemini API (`gemini-2.0-flash-exp`) with quality scoring
- Performs quality assurance with dual threshold checking (transcription confidence and refinement quality)
- Classifies call subjects using local vLLM API (Qwen3-4B) with predefined categories
- Analyzes customer satisfaction using local vLLM API (Qwen3-4B) for sentiment analysis
- Routes low-quality transcripts to manual review automatically
- Ensures all processing runs locally (on-premise) for compliance

**Note**: The correction service is temporarily disabled. The pipeline uses refinement quality scoring to route transcripts directly to classification or manual review.

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
│   ├── correction.py       # Qwen correction service (temporarily disabled)
│   ├── classification.py   # vLLM API classification service (Qwen3-4B)
│   └── sentiment.py        # vLLM API sentiment service (Qwen3-4B)
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
6. **Unified Logging**: All services use consistent logging format with timestamps

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
│   │   ├── correction.py   # Temporarily disabled
│   │   ├── classification.py
│   │   └── sentiment.py
│   │
│   └── pipeline/           # Pipeline orchestration
│       ├── __init__.py
│       └── orchestrator.py
│
└── data/                    # Data directory (excluded from git)
    ├── audio_files/         # Input audio files (structure preserved, contents excluded)
    ├── results/             # Output directory (auto-created)
    └── logs/                # Logs directory (auto-created)
```

**Note**: The `.gitignore` file excludes virtual environments (`cc_agentic_env/`), Python cache files (`__pycache__/`, `*.pyc`), model files, and data directory contents to keep the repository lightweight while preserving directory structure.

## 🚀 Usage

### Basic Usage

```bash
python main.py <audio_path>
```

### Environment Variables

The pipeline supports configuration via environment variables or a `.env` file:

**Required for Refinement:**
- `GEMINI_API_KEY`: Gemini API key for transcript refinement

**Optional for vLLM (Classification & Sentiment):**
- `VLLM_BASE_URL`: vLLM API base URL (default: `http://localhost:8080/v1`)
- `VLLM_MODEL_NAME`: Model name (default: `Qwen/Qwen3-4B`)
- `VLLM_API_KEY`: API key (default: `none`)
- `VLLM_TEMPERATURE`: Temperature for inference (default: `0.1`)

**Example .env file:**
```bash
# .env
GEMINI_API_KEY=your-gemini-api-key-here
VLLM_BASE_URL=http://localhost:8080/v1
VLLM_MODEL_NAME=Qwen/Qwen3-4B
VLLM_TEMPERATURE=0.1
```

The pipeline will automatically load environment variables from the `.env` file if present.

### Prerequisites

1. **vLLM Server**: Ensure your local vLLM server is running at the configured base URL (default: `http://localhost:8080/v1`) with the Qwen3-4B model loaded
2. **Gemini API Key**: Required for transcript refinement (can be skipped, but refinement will fail)

### Programmatic Usage

```python
from src.pipeline import CallAnalysisPipeline
from src.core.config import get_settings

# Get settings
settings = get_settings()

# Optionally customize settings
settings.gemini.api_key = "your-key"
settings.pipeline.confidence_threshold = 0.85
settings.pipeline.refinement_threshold = 0.5
settings.vllm.base_url = "http://localhost:8080/v1"

# Create and run pipeline
pipeline = CallAnalysisPipeline(settings)
result = pipeline.run("path/to/audio.mp3")

# Check final status
print(f"Status: {result['result'].status.value}")
print(f"Refinement Score: {result['result'].refinement_score}")
```

## ⚙️ Configuration

Configuration is managed through `src/core/config.py` using Pydantic models:

### Model Settings

- **WhisperSettings**: Transcription model configuration
  - Default model path: Local path to fine-tuned Whisper model (configured in `config.py`)
  - Supports custom model paths for on-premise deployment
  - Automatic checkpoint detection for fine-tuned models

- **GeminiSettings**: Gemini API configuration for transcript refinement
  - Model used: `gemini-2.0-flash-exp`
  - API key can be set via environment variable or `.env` file
  - Includes refinement prompt template and quality scoring

- **VLLMSettings**: Local vLLM API configuration
  - Base URL: `http://localhost:8080/v1` (configurable)
  - Model: `Qwen/Qwen3-4B` (configurable)
  - Temperature: `0.1` (for classification and sentiment accuracy)
  - Used by both classification and sentiment services

- **LLMSettings**: Local LLM configuration (Qwen) - Currently used for correction service (disabled)

- **ModelSettings**: Base settings for DziriBERT models (legacy, not actively used)

### Pipeline Settings

- `confidence_threshold`: Minimum transcription confidence to proceed (default: 0.90)
- `refinement_threshold`: Minimum refinement score to proceed to classification (default: 0.5)
- `max_retry_attempts`: Maximum correction attempts (default: 2, currently unused)
- `audio_sample_rate`: Target audio sample rate (default: 16000)
- `output_dir`: Results output directory (default: `data/results`)
- `logs_dir`: Logs directory (default: `data/logs`)

### Classification Settings

Fully configurable classification schema:
- Primary categories (default: BILLING & PAYMENTS, ACCOUNT MANAGEMENT, SALES & PRODUCT, COMPLAINT, OTHER)
- Category descriptions
- Sub-categories mapping (e.g., COMPLAINT has 9 sub-categories)
- Default sub-categories
- Other category threshold

## 🔄 Pipeline Flow

The pipeline follows this flow:

```
START
  │
  ▼
[Transcribe] → [Refine] → [Verify]
  │                           │
  │                           ├─[refinement_score < 0.5] → [Save] (MANUAL_REVIEW) → END
  │                           │
  │                           ├─[confidence < 0.9] → [Save] (MANUAL_REVIEW) → END
  │                           │
  │                           └─[Both thresholds pass] → [Classify] → [Sentiment] → [Save] → END
  │
  └─[Error] → [Save] (ERROR) → END
```

### Quality Control

The pipeline uses **dual threshold checking** to ensure quality:

1. **Transcription Confidence** (≥ 0.9): Measures how confident Whisper is in the transcription accuracy
2. **Refinement Score** (≥ 0.5): Measures the meaningfulness and coherence of the refined transcript

**Routing Logic:**
- If `refinement_score < 0.5` → Route to **MANUAL_REVIEW** (transcript has no meaningful content)
- If `confidence_score < 0.9` → Route to **MANUAL_REVIEW** (transcription quality too low)
- If both thresholds pass → Continue to **Classification** and **Sentiment** analysis

This prevents processing meaningless or low-quality transcripts through the classification and sentiment steps.

## 📊 Output Format

Results are saved as JSON files in `data/results/`:

```json
{
  "call_id": "call_audio_abc123",
  "transcript": "Original transcription text...",
  "refined_transcript": "Refined transcription with speaker labels...",
  "confidence_score": 0.95,
  "refinement_score": 0.85,
  "is_corrected": false,
  "subject": "COMPLAINT",
  "sub_subject": "Network Coverage",
  "satisfaction_score": 7.5,
  "status": "COMPLETE",
  "audio_path": "/path/to/audio.mp3",
  "run_count": 1,
  "error_message": null
}
```

### Field Descriptions

- `transcript`: Original transcription from Whisper
- `refined_transcript`: Refined transcript with speaker labels and corrections
- `confidence_score`: Transcription confidence (0.0-1.0)
- `refinement_score`: Refinement quality score (0.0-1.0), indicates transcript coherence and meaning
- `subject`: Primary classification category
- `sub_subject`: Sub-category classification (or "N/A")
- `satisfaction_score`: Customer satisfaction score (0-10, where 0 = not analyzed)
- `status`: Processing status (see Status Values section below)

## 📋 Status Values

The pipeline uses the following status values:

- **PENDING**: Initial state before processing
- **IN_PROGRESS**: Currently being processed
- **COMPLETE**: Successfully processed through all steps
- **MANUAL_REVIEW**: Routed to manual review due to low quality scores
- **LOW_CONFIDENCE**: Low transcription confidence (legacy, currently routes to MANUAL_REVIEW)
- **ERROR**: Error occurred during processing

## 🎨 Service Architecture

### BaseService

All services inherit from `BaseService` which provides:
- Automatic initialization tracking
- Error handling with timing
- Unified logging infrastructure with timestamps
- Consistent interface

### Service Pattern

Each service:
1. Inherits from `BaseService`
2. Implements `initialize()` for model/API loading
3. Implements `process()` for main functionality
4. Uses `_execute_with_timing()` for automatic error handling
5. Logs results in unified format with structured output

### Service Details

**TranscriptionService** (`transcription.py`):
- Uses fine-tuned Whisper model for Arabic/Darija transcription
- Supports automatic chunking for long audio files
- Calculates confidence scores based on transcript length

**RefinementService** (`refinement.py`):
- Uses Gemini API (`gemini-2.0-flash-exp`) for transcript refinement
- Adds speaker labels ([Agent], [Subscriber])
- Fixes Whisper hallucinations and improves punctuation
- Returns quality score (0.0-1.0) indicating transcript coherence

**ClassificationService** (`classification.py`):
- Uses local vLLM API (Qwen3-4B) for zero-shot classification
- Classifies into predefined categories and sub-categories
- Validates responses against configuration
- Returns subject, sub_subject, and confidence

**SentimentService** (`sentiment.py`):
- Uses local vLLM API (Qwen3-4B) for sentiment analysis
- Analyzes customer satisfaction on 1-10 scale
- Returns satisfaction score, sentiment label, and reasoning
- Default score: 0.0 (indicates not analyzed)

**CorrectionService** (`correction.py`):
- Currently disabled
- Would use local Qwen model for transcript correction

### Example Service

```python
class MyService(BaseService):
    def __init__(self, settings):
        super().__init__("my_service")
        self.settings = settings
    
    def initialize(self):
        # Load models, APIs, etc.
        pass
    
    def process(self, input_data):
        def _process():
            # Main processing logic
            return {"result": "..."}
        
        return self._execute_with_timing(_process)
```

## 📝 Logging

All services use a unified logging format:

```
[YYYY-MM-DD HH:MM:SS,mmm] [service.name] [LEVEL] Message
```

### Log Format

- **Timestamp**: ISO format with milliseconds
- **Logger Name**: Service identifier (e.g., `service.transcription`, `pipeline.orchestrator`)
- **Level**: INFO, WARNING, ERROR, DEBUG
- **Message**: Structured log message

### Result Logging

All services log results in a unified format:

```
============================================================
📝 TRANSCRIPTION RESULT
============================================================
[Transcript content]
============================================================

============================================================
✨ REFINEMENT RESULT
============================================================
Status: SUCCESS
Refinement Score: 0.85
Original: ...
Refined:  ...
============================================================

============================================================
📊 CLASSIFICATION RESULT
============================================================
Subject: COMPLAINT
Sub-subject: Network Coverage
Confidence: 0.9500
============================================================

============================================================
😊 SENTIMENT RESULT
============================================================
Satisfaction Score: 7.50/10.0
Sentiment Label: POSITIVE
Confidence: 0.9200
============================================================
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
- **OpenAI** (>=1.0.0) - vLLM API client (for classification and sentiment)
- **Librosa** (0.11.0) - Audio processing
- **Pydantic** (2.12.5) - Configuration and data validation
- **google-genai** - Gemini API client (for transcript refinement)
- **Hugging Face Hub** (0.36.0) - Model repository access
- **Accelerate** (1.12.0) - Model optimization
- **Datasets** (4.4.1) - Data handling utilities

**Note**: The project uses CUDA-enabled PyTorch builds. For CPU-only systems, install the CPU version of PyTorch instead.

## 🛡️ Compliance & Security

- **On-Premise Execution**: All models run locally (Whisper, vLLM)
- **Cloud API**: Only Gemini API is used (for refinement), can be replaced with local alternative
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
- **API Timeouts**: Configurable timeouts for API calls

## 🔍 Troubleshooting

### Refinement Fails

If refinement fails (network error, API key issue):
- Check `GEMINI_API_KEY` is set correctly
- Verify network connectivity
- Pipeline will route to manual review (refinement_score = 0.0)

### vLLM API Not Available

If classification/sentiment fails:
- Ensure vLLM server is running at configured URL
- Check model name matches server configuration
- Pipeline will use fallback values (subject = "UNKNOWN", satisfaction = 0.0)

### Low Quality Transcripts

If transcripts are routed to manual review:
- Check `refinement_score` and `confidence_score` in logs
- Adjust thresholds in `PipelineSettings` if needed
- Review original audio quality

## 📄 License

This implementation is provided as-is for on-premise use in compliance with Algerian data protection laws.
