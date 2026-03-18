from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import logging
import os

# Google AI
from google import genai
from pymongo import MongoClient

# ========================================
# CONFIGURATION
# ========================================

# Environment variables (use .env file or Render dashboard)
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING") or os.getenv("MONGO_URI", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PORT = int(os.getenv("PORT", "8000"))


# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# INITIALIZE SERVICES
# ========================================

# FastAPI app
app = FastAPI(
    title="EduLearn AI Chatbot",
    description="Intelligent chatbot for EduLearn platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize Google AI
gemini_client = None
if GOOGLE_AI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GOOGLE_AI_API_KEY)
        logger.info("✅ Google Gemini AI initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini: {e}")
else:
    logger.warning("⚠️ GOOGLE_AI_API_KEY/GOOGLE_API_KEY not set. API will run in offline-only mode.")

# Initialize MongoDB
mongo_client = None
db = None
chat_history_collection = None
lessons_collection = None

if MONGODB_CONNECTION_STRING:
    try:
        mongo_client = MongoClient(MONGODB_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")
        db = mongo_client['chatbot_db']
        chat_history_collection = db['messages']
        lessons_collection = db['lessons']
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
else:
    logger.warning("⚠️ MONGODB_CONNECTION_STRING/MONGO_URI not set. Database features disabled.")
# ========================================
# DATA MODELS
# ========================================

class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=1, description="Student's question")
    student_id: Optional[str] = Field("anonymous", description="Student ID")
    force_offline: bool = Field(False, description="Force offline mode")

class AnswerResponse(BaseModel):
    """Response model with answer"""
    success: bool
    question: str
    answer: str
    mode: str  # "online" or "offline"
    student_id: str
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ai_status: str
    database_status: str
    timestamp: datetime

# ========================================
# CHATBOT FUNCTIONS
# ========================================

def get_context_from_db(question: str) -> str:
    """
    Fetch relevant lesson content from MongoDB
    Returns context for the question
    """
    if not lessons_collection:
        logger.warning("⚠️ Lessons collection not available")
        return ""
    
     # Use text index if available, else fallback to regex search
        try:
            results = lessons_collection.find(
                {"$text": {"$search": question}},
                {"content": 1, "title": 1, "_id": 0}
            ).limit(3)
        except Exception:
            results = lessons_collection.find(
                {
                    "$or": [
                        {"title": {"$regex": question, "$options": "i"}},
                        {"content": {"$regex": question, "$options": "i"}},
                    ]
                },
                {"content": 1, "title": 1, "_id": 0}
            ).limit(3)
        context_parts = []
        for lesson in results:
            title = lesson.get("title", "")
            content = lesson.get("content", "")
            if content:
                context_parts.append(f"{title}\n{content}")
        
        if context_parts:
            logger.info(f"✅ Found {len(context_parts)} lessons for context")
            return "\n\n".join(context_parts)[:2000]  # Limit context length
        
        logger.info("⚠️ No context found in database")
        return ""

        except Exception as e:
            logger.error(f"❌ Error fetching context: {e}")
            return ""
def generate_online_answer(question: str, context: str) -> Optional[str]:
    """
    Generate answer using Google Gemini AI
    """
    if not gemini_client:
        logger.warning("⚠️ Gemini client not available")
        return None
    
    try:
        # Create prompt with or without context
        if context:
            prompt = f"""You are a helpful tutor for secondary school students.

Answer ONLY using this lesson content:

{context}

Student Question: {question}

Provide a clear, simple answer. If the question cannot be answered from the lesson content, say so politely."""
        else:
            prompt = f"""You are a helpful tutor for secondary school students.

Student Question: {question}

Provide a clear, simple answer suitable for secondary school students."""
        
        # Generate response
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        logger.info("✅ Online answer generated")
        return response.text
    
    except Exception as e:
        logger.error(f"❌ Gemini generation failed: {e}")
        return None

def generate_offline_answer(question: str, context: str) -> str:
    """
    Generate offline answer using only context from database
    """
    if context:
        return (
            "OFFLINE MODE:\n"
            f"I found this from your saved lessons:\n\n{context[:700]}"
        )
    else:
        return "OFFLINE MODE:\nTopic not available in saved lessons. Connect to the internet for more help."

def save_to_history(student_id: str, question: str, answer: str, mode: str):
    """
    Save chat interaction to MongoDB
    """
    if not chat_history_collection:
        return
    
    try:
        chat_history_collection.insert_one({
            "student_id": student_id,
            "question": question,
            "answer": answer,
            "mode": mode,
            "timestamp": datetime.utcnow()
        })
        logger.info(f"✅ Saved to history for student {student_id}")
    except Exception as e:
        logger.error(f"❌ Failed to save history: {e}")

def hybrid_chat(question: str, student_id: str, force_offline: bool = False) -> dict:
    """
    Main hybrid chat function
    Tries online mode first, falls back to offline
    """
    # Step 1: Get context from database
    context = get_context_from_db(question)
    
    # Step 2: Try online mode (unless forced offline)
    mode = "offline"
    answer = None
    
    if not force_offline:
        answer = generate_online_answer(question, context)
        if answer:
            mode = "online"
    
    # Step 3: Fallback to offline mode
    if not answer:
        answer = generate_offline_answer(question, context)
        mode = "offline"
    
    # Step 4: Save to history
    save_to_history(student_id, question, answer, mode)
    
    return {
        "success": True,
        "question": question,
        "answer": answer,
        "mode": mode,
        "student_id": student_id,
        "timestamp": datetime.utcnow()
    }

# ========================================
# API ENDPOINTS
# ========================================

@app.get("/")
async def root():
    """Root endpoint - welcome message"""
    return {
        "message": "EduLearn AI Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "history": "/history/{student_id}",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ai_status = "available" if gemini_client else "unavailable"
    db_status = "connected" if mongo_client else "disconnected"
     overall_status = "healthy" if (gemini_client or mongo_client) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        ai_status=ai_status,
        database_status=db_status,
        timestamp=datetime.utcnow()
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main chatbot endpoint
    
    Ask a question and get an intelligent answer
    """
    logger.info(f"📝 Question from {request.student_id}: {request.question}")
    
    try:
        result = hybrid_chat(
            question=request.question,
            student_id=request.student_id,
            force_offline=request.force_offline
        )
        
        return AnswerResponse(**result)
    
    except Exception as e:
        logger.error(f"❌ Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{student_id}")
async def get_history(student_id: str, limit: int = 50):
    """
    Get chat history for a student
    """
    if not chat_history_collection:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        history = list(chat_history_collection.find(
            {"student_id": student_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        return {
            "success": True,
            "student_id": student_id,
            "count": len(history),
            "history": history
        }
    
    except Exception as e:
        logger.error(f"❌ Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# STARTUP & SHUTDOWN
# ========================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting EduLearn Chatbot API...")
    logger.info(f"✅ AI Status: {'Available' if gemini_client else 'Unavailable'}")
    logger.info(f"✅ Database: {'Connected' if mongo_client else 'Disconnected'}")
    logger.info("✅ API Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down...")
    if mongo_client:
        mongo_client.close()
    logger.info("✅ Cleanup complete")

# ========================================
# RUN APPLICATION
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 EduLearn AI Chatbot API")
    print("=" * 60)
    print(f"📝 Documentation: http://localhost:8000/docs")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print(f"💬 Ask Endpoint: POST http://localhost:8000/ask")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
