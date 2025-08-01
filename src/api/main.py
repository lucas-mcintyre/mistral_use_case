import os
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import evaluation utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from eval_lib.core import KVCache, sha1

# Configuration
DEFAULT_MODEL = 'ft:classifier:ministral-3b-latest:eca5aeb1:20250731:0ba03b81'
CACHE_FILE = "src/api_cache.joblib"
BATCH_SIZE = 32

# Initialize FastAPI app
app = FastAPI(
    title="Mistral Product Classification API",
    description="API for classifying French Amazon products using fine-tuned Mistral model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
client = None
cache = None
model_name = DEFAULT_MODEL

# Pydantic models
class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Product description to classify", min_length=1, max_length=2000)
    use_cache: bool = Field(True, description="Whether to use cached predictions")

class ClassificationResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: Optional[float] = None
    processing_time_ms: float
    cached: bool = False

class BatchClassificationRequest(BaseModel):
    texts: List[str] = Field(..., description="List of product descriptions", min_items=1, max_items=100)
    use_cache: bool = Field(True, description="Whether to use cached predictions")
    limit: Optional[int] = Field(None, description="Maximum number of texts to process (None = all)")

class BatchClassificationResponse(BaseModel):
    predictions: List[ClassificationResponse]
    total_processing_time_ms: float
    average_processing_time_ms: float

class ModelInfo(BaseModel):
    model_name: str
    model_type: str = "fine-tuned-classifier"
    cache_size: int
    cache_file: str
    batch_size: int
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cache_loaded: bool
    timestamp: float

# Initialize function
def initialize_model():
    """Initialize the Mistral client and cache."""
    global client, cache, model_name
    
    # Get API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    # Import Mistral only when needed
    try:
        from mistralai import Mistral
        from mistralai.models import UserMessage
        
        # Initialize client
        client = Mistral(api_key=api_key)
        
        # Initialize cache
        cache_path = Path(CACHE_FILE)
        cache = KVCache(cache_path)
        
        print(f"✅ Model initialized: {model_name}")
        print(f"✅ Cache initialized: {cache_path}")
        
    except ImportError as e:
        print(f"❌ Failed to import Mistral: {e}")
        raise e

# Classification function
def classify_single(text: str, use_cache: bool = True) -> ClassificationResponse:
    """Classify a single product description."""
    global client, cache, model_name
    
    if not client or not cache:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    start_time = time.time()
    
    # Check cache first
    if use_cache:
        cache_key = sha1(text)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            processing_time = (time.time() - start_time) * 1000
            return ClassificationResponse(
                text=text,
                predicted_label=cached_result,
                processing_time_ms=processing_time,
                cached=True
            )
    
    try:
        # Make API call
        response = client.classifiers.classify(
            model=model_name,
            inputs=[text]
        )
        
        # Extract prediction
        result = response.results[0]
        if hasattr(result, 'leaf_path') and hasattr(result.leaf_path, 'scores'):
            # Get the label with highest score
            predicted_label = max(result.leaf_path.scores.items(), key=lambda x: x[1])[0]
            confidence = max(result.leaf_path.scores.values())
        else:
            predicted_label = str(result)
            confidence = None
        
        # Cache the result
        if use_cache:
            cache_key = sha1(text)
            cache.set(cache_key, predicted_label)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResponse(
            text=text,
            predicted_label=predicted_label,
            confidence=confidence,
            processing_time_ms=processing_time,
            cached=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        initialize_model()
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        print("⚠️ API will start but classification endpoints will not work")
        print("💡 Make sure MISTRAL_API_KEY is set and mistralai is installed")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Mistral Product Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=client is not None,
        cache_loaded=cache is not None,
        timestamp=time.time()
    )

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if not cache:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    return ModelInfo(
        model_name=model_name,
        model_type="fine-tuned-classifier",
        cache_size=len(cache.mem),
        cache_file=CACHE_FILE,
        batch_size=BATCH_SIZE,
        status="ready"
    )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_product(request: ClassificationRequest):
    """Classify a single product description."""
    try:
        return classify_single(request.text, request.use_cache)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/limited", response_model=List[ClassificationResponse])
async def classify_limited(texts: List[str], limit: int = 10, use_cache: bool = True):
    """Classify a limited number of product descriptions."""
    if not client or not cache:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Apply limit
    texts_to_process = texts[:limit]
    print(f"[LIMITED] Processing {len(texts_to_process)} texts (limited from {len(texts)})")
    
    start_time = time.time()
    predictions = []
    
    try:
        for text in texts_to_process:
            prediction = classify_single(text, use_cache)
            predictions.append(prediction)
        
        total_time = (time.time() - start_time) * 1000
        print(f"[LIMITED] Completed in {total_time:.2f}ms")
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Limited classification failed: {str(e)}")

@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchClassificationRequest):
    """Classify multiple product descriptions in batch."""
    if not client or not cache:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Apply limit if specified
    texts_to_process = request.texts
    if request.limit is not None:
        texts_to_process = request.texts[:request.limit]
        print(f"[BATCH] Processing {len(texts_to_process)} texts (limited from {len(request.texts)})")
    else:
        print(f"[BATCH] Processing all {len(texts_to_process)} texts")
    
    start_time = time.time()
    predictions = []
    
    try:
        # Process in batches
        for i in range(0, len(texts_to_process), BATCH_SIZE):
            batch_texts = texts_to_process[i:i + BATCH_SIZE]
            
            # Check cache for batch
            batch_results = []
            texts_to_classify = []
            indices_to_classify = []
            
            for j, text in enumerate(batch_texts):
                if request.use_cache:
                    cache_key = sha1(text)
                    cached_result = cache.get(cache_key)
                    if cached_result is not None:
                        batch_results.append(ClassificationResponse(
                            text=text,
                            predicted_label=cached_result,
                            processing_time_ms=0,
                            cached=True
                        ))
                        continue
                
                texts_to_classify.append(text)
                indices_to_classify.append(j)
            
            # Classify uncached texts
            if texts_to_classify:
                response = client.classifiers.classify(
                    model=model_name,
                    inputs=texts_to_classify
                )
                
                for k, result in enumerate(response.results):
                    if hasattr(result, 'leaf_path') and hasattr(result.leaf_path, 'scores'):
                        predicted_label = max(result.leaf_path.scores.items(), key=lambda x: x[1])[0]
                        confidence = max(result.leaf_path.scores.values())
                    else:
                        predicted_label = str(result)
                        confidence = None
                    
                    # Cache the result
                    if request.use_cache:
                        cache_key = sha1(texts_to_classify[k])
                        cache.set(cache_key, predicted_label)
                    
                    batch_results.append(ClassificationResponse(
                        text=texts_to_classify[k],
                        predicted_label=predicted_label,
                        confidence=confidence,
                        processing_time_ms=0,
                        cached=False
                    ))
            
            predictions.extend(batch_results)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(texts_to_process)
        
        return BatchClassificationResponse(
            predictions=predictions,
            total_processing_time_ms=total_time,
            average_processing_time_ms=avg_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

@app.delete("/cache")
async def clear_cache():
    """Clear the prediction cache."""
    global cache
    if cache:
        cache.mem.clear()
        cache_path = Path(CACHE_FILE)
        if cache_path.exists():
            cache_path.unlink()
        print("🗑️ Cache cleared")
        return {"message": "Cache cleared successfully"}
    else:
        raise HTTPException(status_code=500, detail="Cache not initialized")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    if not cache:
        raise HTTPException(status_code=500, detail="Cache not initialized")
    
    return {
        "cache_size": len(cache.mem),
        "cache_file": CACHE_FILE,
        "cache_exists": Path(CACHE_FILE).exists()
    }

# Example usage endpoint
@app.get("/examples")
async def get_examples():
    """Get example product descriptions for testing."""
    examples = [
        "Samsung Galaxy S21 Smartphone avec 128GB de stockage, écran 6.2 pouces, appareil photo 64MP",
        "Sony WH-1000XM4 Casque sans fil avec réduction de bruit, Bluetooth, 30h d'autonomie",
        "Apple MacBook Pro 13 pouces, processeur M1, 8GB RAM, 256GB SSD, macOS",
        "Nike Air Max 270 Chaussures de sport, amorti Air Max, semelle en caoutchouc",
        "Dell XPS 15 Ordinateur portable 15.6 pouces, Intel i7, 16GB RAM, 512GB SSD"
    ]
    
    return {
        "examples": examples,
        "count": len(examples),
        "description": "Example French product descriptions for testing the classification API"
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 