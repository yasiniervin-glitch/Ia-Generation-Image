backend/server.py
from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class ImageGenerationRequest(BaseModel):
    prompt: str

class ImageGenerationResponse(BaseModel):
    id: str
    prompt: str
    image_base64: str
    created_at: datetime

class GeneratedImage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    image_base64: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Image Generation API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """Generate an image from a text prompt using OpenAI GPT-Image-1"""
    try:
        from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration
        
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        logger.info(f"Generating image for prompt: {request.prompt}")
        
        # Initialize the image generator
        image_gen = OpenAIImageGeneration(api_key=api_key)
        
        # Generate images
        images = await image_gen.generate_images(
            prompt=request.prompt,
            model="gpt-image-1",
            number_of_images=1
        )
        
        if images and len(images) > 0:
            # Convert image to base64
            image_base64 = base64.b64encode(images[0]).decode('utf-8')
            
            # Save to database
            generated_image = GeneratedImage(
                prompt=request.prompt,
                image_base64=image_base64
            )
            await db.generated_images.insert_one(generated_image.dict())
            
            logger.info(f"Image generated successfully for prompt: {request.prompt}")
            
            return ImageGenerationResponse(
                id=generated_image.id,
                prompt=generated_image.prompt,
                image_base64=image_base64,
                created_at=generated_image.created_at
            )
        else:
            raise HTTPException(status_code=500, detail="No image was generated")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

@api_router.get("/images", response_model=List[ImageGenerationResponse])
async def get_generated_images():
    """Get all previously generated images"""
    try:
        images = await db.generated_images.find().sort("created_at", -1).to_list(50)
        return [
            ImageGenerationResponse(
                id=img["id"],
                prompt=img["prompt"],
                image_base64=img["image_base64"],
                created_at=img["created_at"]
            )
            for img in images
        ]
    except Exception as e:
        logger.error(f"Error fetching images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch images: {str(e)}")

@api_router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """Delete a generated image"""
    try:
        result = await db.generated_images.delete_one({"id": image_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"message": "Image deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
