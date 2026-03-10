# api.py - This file creates a web API for your HTS matcher

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import os
from typing import Optional, List
import logging

# Import your existing HTS matcher
from hts_matcher import HTSMatcherEmbeddingsLocal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request/response models
class MatchRequest(BaseModel):
    description: str
    top_n: int = 5

class MatchResult(BaseModel):
    hts_number: str
    score: float
    full_description: str
    readable_description: str

class MatchResponse(BaseModel):
    success: bool
    matches: List[MatchResult]
    query: str
    message: Optional[str] = None

# Create FastAPI app
app = FastAPI(title="HTS Matcher API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for matcher
matcher = None

@app.on_event("startup")
async def load_matcher():
    """Load the HTS matcher when the API starts"""
    global matcher
    try:
        data_path = Path("hts_1_97_stacked.csv")
        logger.info(f"Loading HTS matcher from {data_path}")
        matcher = HTSMatcherEmbeddingsLocal(data_path)
        logger.info("✅ HTS matcher loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load HTS matcher: {e}")

@app.get("/health")
async def health_check():
    if matcher is None:
        return {"status": "error", "message": "Matcher not loaded"}
    return {"status": "healthy", "message": "API is running"}

@app.post("/match", response_model=MatchResponse)
async def match_hts(request: MatchRequest):
    if matcher is None:
        raise HTTPException(status_code=503, detail="Matcher not loaded")
    
    try:
        logger.info(f"🔍 Matching: {request.description}")
        
        results_df = matcher.match(
            query=request.description, 
            top_n=request.top_n, 
            interactive=False
        )
        
        if results_df is None or (hasattr(results_df, 'empty') and results_df.empty):
            return MatchResponse(
                success=False,
                matches=[],
                query=request.description,
                message="No matches found"
            )
        
        matches = []
        if hasattr(results_df, 'iterrows'):
            for _, row in results_df.iterrows():
                matches.append(MatchResult(
                    hts_number=str(row.get('HTS_Number', 'N/A')),
                    score=float(row['score']),
                    full_description=str(row.get('Full_Description', '')),
                    readable_description=str(row.get('Readable_Description', ''))
                ))
        else:
            matches.append(MatchResult(
                hts_number=str(results_df.get('HTS_Number', 'N/A')),
                score=float(results_df['score']),
                full_description=str(results_df.get('Full_Description', '')),
                readable_description=str(results_df.get('Readable_Description', ''))
            ))
        
        return MatchResponse(
            success=True,
            matches=matches[:request.top_n],
            query=request.description,
            message=f"Found {len(matches)} matches"
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)