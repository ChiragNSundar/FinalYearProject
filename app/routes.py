import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from app.video_processing import process_video
from app.db import read_violations_by_vehicle
import shutil

router = APIRouter()

@router.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process video and detect violations
        process_video(file_path)
        return {"message": "Video processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-violations/")
async def get_violations(vehicle_number: str = Form(...)):
    try:
        violations = read_violations_by_vehicle(vehicle_number)
        return JSONResponse(content=violations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))