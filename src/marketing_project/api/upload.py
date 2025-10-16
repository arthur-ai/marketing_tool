"""
File upload API endpoints for Marketing Project.
"""

import logging
import os
import uuid
import json
import csv
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

# Document processing imports
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from marketing_project.services.content_source_factory import ContentSourceManager
from marketing_project.services.content_source_config_loader import ContentSourceConfigLoader

logger = logging.getLogger("marketing_project.api.upload")

# Create router
router = APIRouter()

# Initialize content source manager
content_manager = ContentSourceManager()
config_loader = ContentSourceConfigLoader()

# Upload directories
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CONTENT_DIR = os.getenv("CONTENT_DIR", "content")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONTENT_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    content_type: str = Form("blog_post")
):
    """
    Upload a file and process it for the marketing pipeline.
    
    Args:
        file: The file to upload
        content_type: Type of content (blog_post, transcript, release_notes)
    """
    try:
        # Validate file type
        allowed_extensions = {
            '.json', '.md', '.txt', '.yaml', '.yml', 
            '.docx', '.doc', '.pdf', '.csv', '.rtf'
        }
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Validate file size (25MB limit)
        max_size = 25 * 1024 * 1024  # 25MB
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 25MB"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        
        # Save to uploads directory
        upload_path = os.path.join(UPLOAD_DIR, safe_filename)
        with open(upload_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"File uploaded: {safe_filename} ({len(content)} bytes)")
        
        # Process file based on content type
        processed_path = await process_uploaded_file(upload_path, content_type, file_ext)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "File uploaded and processed successfully",
                "file_id": file_id,
                "filename": file.filename,
                "size": len(content),
                "content_type": content_type,
                "upload_path": upload_path,
                "processed_path": processed_path
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    if not DOCX_AVAILABLE:
        raise HTTPException(status_code=500, detail="DOCX processing not available")
    
    try:
        doc = DocxDocument(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process DOCX file: {str(e)}")


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    if not PDF_AVAILABLE:
        raise HTTPException(status_code=500, detail="PDF processing not available")
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return '\n'.join(text)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {str(e)}")


def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
            
            # Convert CSV to readable text format
            text_lines = []
            for i, row in enumerate(rows):
                if i == 0:  # Header row
                    text_lines.append("Headers: " + " | ".join(row))
                else:
                    text_lines.append("Row " + str(i) + ": " + " | ".join(row))
            
            return '\n'.join(text_lines)
    except Exception as e:
        logger.error(f"Failed to extract text from CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


def extract_text_from_rtf(file_path: str) -> str:
    """Extract text from RTF file (basic implementation)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Basic RTF text extraction (removes RTF formatting codes)
            import re
            # Remove RTF control words and braces
            text = re.sub(r'\\[a-z]+\d*\s?', '', content)
            text = re.sub(r'[{}]', '', text)
            return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from RTF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process RTF file: {str(e)}")


async def process_uploaded_file(upload_path: str, content_type: str, file_ext: str) -> str:
    """
    Process an uploaded file and move it to the appropriate content directory.
    """
    try:
        # Create content directory structure
        content_type_dir = os.path.join(CONTENT_DIR, f"{content_type}s")
        os.makedirs(content_type_dir, exist_ok=True)
        
        # Generate processed filename
        filename = Path(upload_path).name
        processed_path = os.path.join(content_type_dir, filename)
        
        # Extract text content based on file type
        if file_ext == '.json':
            # Read JSON file
            with open(upload_path, 'r', encoding='utf-8') as f:
                content_data = f.read()
        elif file_ext in ['.docx', '.doc']:
            # Extract text from Word document
            content_data = extract_text_from_docx(upload_path)
        elif file_ext == '.pdf':
            # Extract text from PDF
            content_data = extract_text_from_pdf(upload_path)
        elif file_ext == '.csv':
            # Extract text from CSV
            content_data = extract_text_from_csv(upload_path)
        elif file_ext == '.rtf':
            # Extract text from RTF
            content_data = extract_text_from_rtf(upload_path)
        else:
            # Read text files (md, txt, yaml, yml)
            with open(upload_path, 'r', encoding='utf-8') as f:
                content_data = f.read()
        
        # Process based on file type
        if file_ext == '.json':
            # Validate JSON structure
            import json
            try:
                json_data = json.loads(content_data)
                # Ensure required fields for content
                if 'title' not in json_data:
                    json_data['title'] = Path(filename).stem
                if 'content' not in json_data:
                    json_data['content'] = content_data
                if 'id' not in json_data:
                    json_data['id'] = str(uuid.uuid4())
                
                # Write processed JSON
                with open(processed_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
            except json.JSONDecodeError:
                # If not valid JSON, create a simple structure
                json_data = {
                    "id": str(uuid.uuid4()),
                    "title": Path(filename).stem,
                    "content": content_data,
                    "type": content_type,
                    "created_at": str(uuid.uuid4())  # Placeholder for timestamp
                }
                with open(processed_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        elif file_ext in ['.md', '.txt', '.docx', '.doc', '.pdf', '.csv', '.rtf', '.yaml', '.yml']:
            # For all other files, create a simple structure
            json_data = {
                "id": str(uuid.uuid4()),
                "title": Path(filename).stem,
                "content": content_data,
                "type": content_type,
                "original_format": file_ext,
                "created_at": str(uuid.uuid4())  # Placeholder for timestamp
            }
            
            # Save as JSON for consistency
            json_path = processed_path.replace(file_ext, '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            processed_path = json_path
        
        else:
            # For other file types, just copy
            import shutil
            shutil.copy2(upload_path, processed_path)
        
        logger.info(f"File processed: {upload_path} -> {processed_path}")
        
        # Clean up upload file
        try:
            os.remove(upload_path)
        except OSError:
            logger.warning(f"Could not remove upload file: {upload_path}")
        
        return processed_path
        
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )


@router.get("/upload/status/{file_id}")
async def get_upload_status(file_id: str):
    """
    Get the status of an uploaded file.
    """
    try:
        # This would typically check a database or cache for file status
        # For now, we'll return a simple response
        return JSONResponse(
            status_code=200,
            content={
                "file_id": file_id,
                "status": "processed",
                "message": "File has been processed successfully"
            }
        )
    except Exception as e:
        logger.error(f"Failed to get upload status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get upload status: {str(e)}"
        )
