import io
import os
import base64
import decord
import tempfile
import soundfile as sf
import torch
import openl3
import whisper
import librosa
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from docx import Document
from PyPDF2 import PdfReader
from pptx import Presentation
from transformers import logging
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer
from moviepy.video.io.VideoFileClip import VideoFileClip

logging.set_verbosity_error()  # Reduce unneeded logs

plain_text_types = [
        "text/plain",                      # TXT
        "application/json",                 # JSON
        "application/xml", "text/xml",      # XML
        "text/html",                        # HTML
        "text/css",                         # CSS
        "text/javascript",                   # JavaScript
        "application/javascript",            # JavaScript (alternativo)
        "application/x-httpd-php",           # PHP
        "application/x-sh",                  # Shell script
        "application/x-python-code",         # Python
        "text/markdown",                     # Markdown
        "application/sql",                   # SQL scripts
        "text/csv",                          # CSV
        "application/x-yaml", "text/yaml",   # YAML
        "text/x-c", "text/x-c++",            # C y C++
        "text/x-java-source",                # Java
        "text/x-go",                         # Go
        "text/x-ruby",                       # Ruby
        "text/x-perl",                       # Perl
        "text/x-php",                        # PHP
        "text/x-shellscript"                 # Shell scripts (.sh)
    ]

image_types = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff"
    ]

class UniversalEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Attempting to load embedding models into {self.device}")
        # Load models
        # Text model
        self.text_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4", device=self.device)
        # Audio model (size 512 instead of 8192)
        self.audio_model = openl3.models.load_audio_embedding_model(
                input_repr="mel256",
                content_type="env",
                embedding_size=512,
                frontend='librosa'
            )
        self.audio_model = self.audio_model.to(self.device)
        # Image model
        self.image_model = openl3.models.load_audio_embedding_model(
                input_repr="image",
                content_type="objects",
                embedding_size=512
            )
        self.image_model = self.image_model.to(self.device)
        self.audio_sample_rate = 48000  # OpenL3 uses 48 kHz
        self.tr_model = whisper.load_model("small", device=self.device)
        self.tesseract_path = "/usr/bin/tesseract"  # Path to tesseract
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        

    def embed(self, data, storable = False):
        """
            Arguments:
                data(json request params):
                    - input: plain text or base64
            Returns:
                object(images do not return raw):
                    - emb: [0,1,2,3],
                    - raw: ["paragraph1", "paragraph1"]
        """
        
        try:
            # Define text or base64
            if not data["input"].startswith("data:") and ";base64," not in data["input"]:
                data["text"] = data["input"]
            # Plain text should come in text property
            if "text" in data:
                return self.embed_text(data["text"], storable)
            else:
                # Retrieve base64 header
                header, content = data["input"].split(",")
                header = header.split(":")[1].split(";")[0]
                # Retrieve base64 content
                print(header)
                decoded_data = base64.b64decode(content)
                
                # Plain text types
                if header in plain_text_types:
                    print("Plain text file (TXT, JSON, XML, HTML, JS, CSS, etc.)")
                    text = decoded_data.decode("utf-8")  # Get content
                    return self.embed_text(text, storable), None
                    
                # PDF document types
                elif "application/pdf" in header:
                    print("PDF file")
                    chunks = []
                    with io.BytesIO(decoded_data) as pdf_buffer:
                        reader = PdfReader(pdf_buffer)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                page_text = page_text.replace("\n", "")
                                chunks.append(page_text)
                    
                    # Convert PDF pages to images for OCR
                    if len(chunks) == 0:
                        print("Applying OCR to scanned PDF")
                        images = convert_from_bytes(decoded_data, dpi=300)
                        for image in images:
                            ocr_text = self.ocr_image(image).strip()
                            chunks.append(ocr_text)
                    
                    # Build results
                    results = { "bytes": [], "raw": [] }
                    for chunk in chunks:
                        ch_res, error = self.embed_text(chunk, storable)
                        if error is None:
                            for r, b in zip(ch_res["raw"], ch_res["bytes"]):
                                results["raw"].append(r)
                                results["bytes"].append(b)
                                
                    return results, None

                # Word (DOCX)
                elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in header:
                    print("Word File (DOCX)")
                    
                    # Read document
                    with io.BytesIO(decoded_data) as doc_buffer:
                        doc = Document(doc_buffer)  # Load .docx file
                        chunks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]  # Extract non empty paragraphs
                    
                    # Build results
                    results = { "bytes": [], "raw": [] }
                    for chunk in chunks:
                        ch_res, error = self.embed_text(chunk, storable)
                        if error is None:
                            for r, b in zip(ch_res["raw"], ch_res["bytes"]):
                                results["raw"].append(r)
                                results["bytes"].append(b)
                    
                    return results, None

                # Excel (XLSX, XLS)
                elif header in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                    print("Excel file (XLSX, XLS)")
                    
                    with io.BytesIO(decoded_data) as file_buffer:
                        xls = pd.ExcelFile(file_buffer)  # Load the Excel file
                        chunks = [xls.parse(sheet).to_string(index=False) for sheet in xls.sheet_names]  # Convert each sheet to a string
                    
                    
                    # Build results
                    results = { "bytes": [], "raw": [] }
                    for chunk in chunks:
                        ch_res, error = self.embed_text(chunk, storable)
                        if error is None:
                            for r, b in zip(ch_res["raw"], ch_res["bytes"]):
                                results["raw"].append(r)
                                results["bytes"].append(b)
                                
                    return results, None

                # PowerPoint (PPTX, PPT)
                elif header in ["application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/vnd.ms-powerpoint"]:
                    print("PowerPoint file (PPTX, PPT)")
                    
                    with io.BytesIO(decoded_data) as file_buffer:
                        presentation = Presentation(file_buffer)  # Load the PowerPoint file
                        chunks = []
                        
                        for slide in presentation.slides:
                            slide_text = "\n".join([shape.text.strip() for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()])
                            if slide_text:
                                chunks.append(slide_text)  # Each slide is a separate chunk
                    
                    
                    # Build results
                    results = { "bytes": [], "raw": [] }
                    for chunk in chunks:
                        ch_res, error = self.embed_text(chunk, storable)
                        if error is None:
                            for r, b in zip(ch_res["raw"], ch_res["bytes"]):
                                results["raw"].append(r)
                                results["bytes"].append(b)
                                
                    return results, None

                # Imágenes (JPG, PNG, GIF, BMP, TIFF)
                elif header in image_types:
                    print("Image file (JPG, PNG, GIF, BMP, TIFF)")
                    
                    result = { "bytes": [], "raw": [] }
                    
                    with io.BytesIO(decoded_data) as image_buffer:
                        image_embedding, error = self.embed_image(image_buffer)
                        if error is None:
                            result["raw"].append("Protected image data")
                            result["bytes"].append(image_embedding.tolist() if storable else image_embedding)
                        # Attempt to extract text using OCR
                        ocr_text, ocr_err = self.ocr_image(image_buffer).strip()
                        if ocr_err is None:
                            ocr_embedings, ocr_emb_err = self.embed_text(ocr_text, storable)
                            if ocr_emb_err is None:
                                for r, b in zip(ocr_embedings["raw"], ocr_embedings["bytes"]):
                                    result["raw"].append(r)
                                    result["bytes"].append(b)
                    
                    return result, None

                # Audios (MP3, WAV, OGG, AAC)
                elif header in ["audio/mpeg", "audio/wav", "audio/ogg", "audio/aac"]:
                    print("Audio file (MP3, WAV, OGG, AAC)")
                    
                    # Read file
                    with io.BytesIO(decoded_data) as audio_buffer:
                        audio, sr = librosa.load(audio_buffer, sr=self.audio_sample_rate)  # Load the audio file
                    
                    result = { "raw": [], "bytes": [] }
                    # Generate raw audio embeddings
                    audio_embedding, error = self.embed_audio(audio, sr)  # Generate audio embedding
                    if error is None:
                        result["raw"].append("Protected audio data")
                        result["bytes"].append(audio_embedding.tolist() if storable else audio_embedding)

                    # Transcribe speech to text using Whisper
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
                        sf.write(temp_audio.name, audio, sr)  # Save audio as WAV
                        transcribed_text = self.tr_model.transcribe(temp_audio.name)["text"].strip()
                        # Generate transcription embeddings
                        tr_embeddings, tr_err = self.embed_text(transcribed_text, storable)
                        if tr_err is None:
                            for r, b in zip(tr_embeddings["raw"], tr_embeddings["bytes"]):
                                result["raw"].append(r)
                                result["bytes"].append(b)
                    
                    return result, None
                    
                # Videos (MP4, AVI, MKV, WEBM, MOV)
                elif header in ["video/mp4", "video/x-msvideo", "video/x-matroska", "video/webm", "video/quicktime"]:
                    print("Video file (MP4, AVI, MKV, WEBM, MOV)")

                    try:
                        result = { "raw": [], "bytes": [] }
                        
                        # Attempt to read video file
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                            temp_video.write(decoded_data)
                            temp_video.flush()  # Ensure data is written before reading
                            temp_video_path = temp_video.name  # Store file path
                            
                            
                            # Extract 1 frame each second
                            vr = decord.VideoReader(temp_video_path)
                            fps = vr.get_avg_fps() # Get framerate
                            frame_indexes = [int(i * fps) for i in range(int(len(vr) / fps))] # Get frame indexes
                            frames = [vr[i].asnumpy() for i in frame_indexes if i < len(vr)] # Get frames
                            
                            # Generate image embeddings per frame extracted
                            for frame in frames:
                                fr_emb, fr_err = self.embed_image(frame)
                                if fr_err is None:
                                    result["raw"].append("Protected image data")
                                    result["bytes"].append(fr_emb.tolist() if storable else fr_emb)
                            
                            # Extract audio embeddings
                            audio_path = temp_video_path.replace(".mp4", ".wav")
                            video = VideoFileClip(temp_video_path)
                            video.audio.write_audiofile(audio_path, codec="pcm_s16le", fps=self.audio_sample_rate)
                            
                            # Generate audio embeddings for extracted audio
                            audio_emb, audio_err = self.embed_audio(audio, sr)
                            if audio_err is None:
                                result["raw"].append("Protected audio data")
                                result["bytes"].append(audio_emb.tolist() if storable else audio_emb)
                            
                            # Extract text transcription for audio
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
                                sf.write(temp_audio.name, audio, sr)
                                transcribed_text = self.tr_model.transcribe(temp_audio.name)["text"].strip()

                                # Generate text embeddings for audio transcription
                                text_emb, text_err = self.embed_text(transcribed_text, storable)
                                if text_err is None:
                                    for r, b in zip(text_emb["raw"], text_emb["bytes"]):
                                        result["raw"].append(r)
                                        result["bytes"].append(b)

                        return result, None
                    finally:
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                        if os.path.exists(audio_path):
                            os.remove(audio_path)

                # Tipo desconocido
                else:
                    print("Unknown file type")
                    return {
                        "status": "error",
                        "message": "Unknown file type"
                    }
                    
        except Exception as e:
            print(f"Error processing file: {e}")
            return None, e

    def embed_text(self, text, storable):
        """Generate text embeddings"""
        try:
            chunks = text.split("\n")
            response = {
                "bytes": [],
                "raw": []
            }
            for chunk in chunks:
                chunk = chunk.strip()  # remove spaces
                if chunk:  # Validate non empty string
                    response["raw"].append(chunk)
                    response["bytes"].append(self.text_model.encode(chunk).tolist() if storable else self.text_model.encode(chunk))
            return response, None
        except Exception as err:
            print(f"Failed to generate text embeddings due to: {err}")
            return None, err

    def embed_image(self, image_bytes):
        """Generate image embeddings using OpenL3."""
        try:
            # Process image
            image = Image.open(image_bytes).convert("RGB")
            image = image.resize((224, 224))
            image = np.array(image).astype(np.float32) / 255.0  # Normalise
            img_batch = np.expand_dims(image, axis=0)
            # Request embeddings
            with torch.no_grad():
                emb = self.image_model(torch.from_numpy(img_batch).permute(0,3,1,2).to(self.device))
            # Retrieve embeddings
            emb_result = emb.cpu().numpy()
            return emb_result.flatten(), None
        except Exception as err:
            print(f"Failed to generate image embeddings due to: {err}")
            return None, err

    def embed_audio(self, audio_bytes, sr):
        """Generate audio embeddings using OpenL3."""
        try:
            with torch.no_grad():
                emb = openl3.get_audio_embedding(audio_bytes, sr, model=self.audio_model, content_type="env")
                return emb.flatten(), None
            
        except Exception as err:
            print(f"Failed to generate audio embeddings due to: {err}")
            return None, err
            

    def ocr_image(self, image_bytes):
        """Apply OCR to provided image"""
        try:
            img = Image.open(image_bytes)
            return pytesseract.image_to_string(img), None
        except Exception as err:
            print(f"Failed to OCR image due to: {err}")
            return None, err
            
