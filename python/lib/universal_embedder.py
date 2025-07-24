import base64
import io
import os
import tempfile
import soundfile as sf
import torch
import librosa
import whisper
import openl3
from moviepy.video.io.VideoFileClip import VideoFileClip
import decord
import numpy as np
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from transformers import logging
import pandas as pd

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
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
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
        self.tesseract_path = "/usr/bin/tesseract"  # Path to tesseract
        self.tr_model = whisper.load_model("small")
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
                    if header == "text/plain":  # TXT files -> Split in chunks
                        return self.embed_text(text, storable)
                    # Process whole file content
                    chunks = [text]
                    return {
                        "bytes": [self.embed_text(chunk, storable) for chunk in chunks],
                        "raw": chunks
                    }
                    
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
                    all_results = [self.embed_text(chunk, storable) for chunk in chunks]
                    return {
                        "bytes": [item for sublist in all_results for item in sublist["bytes"]],
                        "raw": [item for sublist in all_results for item in sublist["raw"]]
                    }

                # Word (DOCX)
                elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in header:
                    print("Word File (DOCX)")
                    
                    with io.BytesIO(decoded_data) as doc_buffer:
                        doc = Document(doc_buffer)  # Load .docx file
                        chunks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]  # Extract non empty paragraphs

                    return {
                        "bytes": [self.text_model.encode(chunk).tolist() if storable else self.text_model.encode(chunk) for chunk in chunks],
                        "raw": chunks
                    }

                # Excel (XLSX, XLS)
                elif header in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                    print("Excel file (XLSX, XLS)")
                    
                    with io.BytesIO(decoded_data) as file_buffer:
                        xls = pd.ExcelFile(file_buffer)  # Load the Excel file
                        chunks = [xls.parse(sheet).to_string(index=False) for sheet in xls.sheet_names]  # Convert each sheet to a string
                    
                    return {
                        "bytes": [self.text_model.encode(chunk).tolist() if storable else self.text_model.encode(chunk) for chunk in chunks],
                        "raw": chunks
                    }

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
                    
                    return {
                        "bytes": [self.text_model.encode(chunk).tolist() if storable else self.text_model.encode(chunk) for chunk in chunks],
                        "raw": chunks
                    }

                # Imágenes (JPG, PNG, GIF, BMP, TIFF)
                elif header in image_types:
                    print("Image file (JPG, PNG, GIF, BMP, TIFF)")
                    
                    with io.BytesIO(decoded_data) as image_buffer:
                        image_embedding = self.embed_image(image_buffer)
                        if image_embedding is None:
                            return None
                        # Extract text using OCR
                        ocr_text = self.ocr_image(image_buffer).strip()
                    
                    chunks = []
                    embeddings = []
                    
                    # Store the image embedding as a chunk
                    chunks.append("Image embedding")
                    embeddings.append(image_embedding.tolist() if storable else image_embedding)

                    # Store OCR text as a separate chunk if text was found
                    if ocr_text:
                        chunks.append(ocr_text)
                        embeddings.append(self.text_model.encode(ocr_text).tolist() if storable else self.text_model.encode(ocr_text))
                    
                    return {
                        "bytes": embeddings,
                        "raw": [data]
                    }

                # Audios (MP3, WAV, OGG, AAC)
                elif header in ["audio/mpeg", "audio/wav", "audio/ogg", "audio/aac"]:
                    print("Audio file (MP3, WAV, OGG, AAC)")
                    
                    with io.BytesIO(decoded_data) as audio_buffer:
                        audio, sr = librosa.load(audio_buffer, sr=self.audio_sample_rate)  # Load the audio file
                        audio_embedding = openl3.get_audio_embedding(audio, sr, content_type="env").flatten()  # Generate audio embedding

                    # Transcribe speech to text using Whisper
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
                        sf.write(temp_audio.name, audio, sr)  # Save audio as WAV
                        transcribed_text = self.tr_model.transcribe(temp_audio.name)["text"].strip()

                    chunks = []
                    embeddings = []

                    # Store the audio embedding as a chunk
                    chunks.append("Audio embedding")
                    embeddings.append(audio_embedding.tolist() if storable else audio_embedding)

                    # Store transcribed text as a separate chunk if speech was detected
                    if transcribed_text.strip():
                        chunks.append(transcribed_text)
                        embeddings.append(self.text_model.encode(transcribed_text).tolist() if storable else self.text_model.encode(transcribed_text))
                    
                    return {
                        "bytes": embeddings,
                        "raw": chunks
                    }
                    
                # Videos (MP4, AVI, MKV, WEBM, MOV)
                elif header in ["video/mp4", "video/x-msvideo", "video/x-matroska", "video/webm", "video/quicktime"]:
                    print("Video file (MP4, AVI, MKV, WEBM, MOV)")

                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                        temp_video.write(decoded_data)
                        temp_video.flush()  # Ensure data is written before reading
                        temp_video_path = temp_video.name  # Store file path

                        try:
                            # Extract audio
                            video = VideoFileClip(temp_video_path)
                            audio_path = temp_video_path.replace(".mp4", ".wav")
                            video.audio.write_audiofile(audio_path, codec="pcm_s16le", fps=self.audio_sample_rate)

                            # Extract frames for video embedding
                            vr = decord.VideoReader(temp_video_path)
                            frames = [vr[i].asnumpy() for i in range(0, len(vr), max(1, len(vr)//10))]  # Sample 10 frames

                            # Generate video embedding
                            video_embedding = np.mean(
                                [openl3.get_image_embedding(frame, content_type="env").flatten() for frame in frames], axis=0
                            )

                        finally:
                            os.remove(temp_video_path)
                            if 'audio_path' in locals() and os.path.exists(audio_path):
                                os.remove(audio_path)


                    # Generate audio embedding & transcription
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
                        audio_data, _ = librosa.load(audio_path, sr=self.audio_sample_rate)
                        sf.write(temp_audio.name, audio_data, self.audio_sample_rate)
                        transcribed_text = self.tr_model.transcribe(temp_audio.name)["text"].strip()
                        audio_embedding = openl3.get_audio_embedding(audio_data, self.audio_sample_rate, content_type="env").flatten()

                    chunks = ["Video embedding"]
                    embeddings = [video_embedding.tolist() if storable else video_embedding]

                    # Store audio embedding
                    chunks.append("Audio embedding")
                    embeddings.append(audio_embedding.tolist() if storable else audio_embedding)

                    # Store transcribed text if speech is detected
                    if transcribed_text.strip():
                        chunks.append(transcribed_text)
                        embeddings.append(self.text_model.encode(transcribed_text).tolist() if storable else self.text_model.encode(transcribed_text))

                    return {
                        "bytes": embeddings,
                        "raw": chunks
                    }

                # Tipo desconocido
                else:
                    print("Unknown file type")
                    return {
                        "status": "error",
                        "message": "Unknown file type"
                    }
                    
        except Exception as e:
            print(f"Error processing file: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def embed_text(self, text, storable):
<<<<<<< HEAD
        try:
            """Generate text embeddings"""
=======
        """Generate text embeddings"""
        try:
>>>>>>> cbb492d73457c86c3d5c7b348da199c50926a88b
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
            return response
        except Exception as err:
<<<<<<< HEAD
            print(f"Erron in embed_text(): {err}")
=======
            print(f"Failed to generate text embeddings due to: {err}")
>>>>>>> cbb492d73457c86c3d5c7b348da199c50926a88b
            return None

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
            return emb_result.flatten()
        except Exception as err:
            print(f"Failed to generate image embeddings due to: {err}")
            return None

    def embed_audio(self, audio_bytes, sr):
        """Generate audio embeddings using OpenL3."""
        try:
            with torch.no_grad():
                emb = openl3.get_audio_embedding(audio_bytes, sr, model=self.audio_model, content_type="env")
                return emb.flatten()
            
        except Exception as err:
            print(f"Failed to generate audio embeddings due to: {err}")
            return None
            

    def ocr_image(self, image_bytes):
<<<<<<< HEAD
        try:
            """Apply OCR to provided PIL Image object"""
            if image_bytes.mode not in ['L', 'RGB']:
                image_bytes = image_bytes.convert('RGB')
                print(f"Imagen convertida a modo RGB para OCR.")
            
            extracted_text = pytesseract.image_to_string(image_bytes)
            return extracted_text
        except Exception as err:
            print(err)
            return ""
=======
        """Apply OCR to provided image"""
        img = Image.open(image_bytes)
        return pytesseract.image_to_string(img)
>>>>>>> cbb492d73457c86c3d5c7b348da199c50926a88b
