from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re
from dotenv import load_dotenv
import openai
from pydantic import BaseModel
from typing import Optional

# Load environment variables
load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# 메뉴 아이템과 가격 (원 단위)
menu_items = {
    "돼지국밥": 9000,
    "순대국밥": 10000,
    "내장국밥": 9500,
    "섞어국밥": 9500,
    "수육 반접시": 13000,
    "수육 한접시": 25000
}
menu_list = list(menu_items.keys())

class OrderRequest(BaseModel):
    audio_data: str  # base64 encoded audio data

class CorrectedOrder(BaseModel):
    original_text: str
    corrected_text: str
    order_items: dict

class VoiceCommand(BaseModel):
    text: str

# Korean number mapping
KOREAN_NUM = {
    "한": 1, "하나": 1, "않아": 1, "아나": 1, "일": 1,
    "두": 2, "둘": 2, "이": 2,
    "세": 3, "셋": 3, "삼": 3,
    "네": 4, "넷": 4, "사": 4,
    "다섯": 5, "오": 5,
    "여섯": 6, "육": 6,
    "일곱": 7, "칠": 7,
    "여덟": 8, "팔": 8,
    "아홉": 9, "구": 9,
    "열": 10, "십": 10
}

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def correct_text_with_gpt(text: str) -> str:
    """Use GPT to correct the recognized text for food ordering"""
    try:
        system_prompt = """
        당신은 한국어 음성 주문을 정확한 메뉴명과 수량으로 변환하는 전문가입니다.
        다음 메뉴 중에서 가장 유사한 메뉴로 교정하고, 수량을 정확히 인식해주세요:
        
        [메뉴 목록]
        1. 돼지국밥
        2. 내장국밥
        3. 섞어국밥
        4. 순대국밥
        5. 수육 한접시
        6. 수육 반접시

        [주문 예시]
        - "섞어 5개" → "섞어국밥 5개"
        - "내장 세 그릇" → "내장국밥 3개"
        - "돼지 둘" → "돼지국밥 2개"
        - "순대 하나" → "순대국밥 1개"
        - "수육 한접시 셋" → "수육 한접시 3개"
        - "수육 반 두 개" → "수육 반접시 2개"

        [지시사항]
        1. 수량은 반드시 보존하고, 누락되지 않도록 하세요.
        2. 수량 표현은 다음과 같이 변환하세요:
           - "하나", "한 개", "한 그릇", "일" → "1개"
           - "둘", "두 개", "두 그릇", "이" → "2개"
           - "셋", "세 개", "세 그릇", "서", "삼" → "3개"
           - "넷", "네 개", "네 그릇", "사" → "4개"
           - "다섯", "다섯 개", "오" → "5개"
           - "여섯", "육" → "6개"
           - "일곱", "칠" → "7개"
           - "여덟", "팔" → "8개"
           - "아홉", "구" → "9개"
           - "열", "십" → "10개"
        3. 메뉴 이름이 생략되었거나 줄여서 말해도 정확한 메뉴명으로 변환하세요.
        4. 최종 출력 형식은 반드시 "메뉴이름 수량" 형식으로 하세요.
        5. 수량이 없으면 "1개"로 기본값을 설정하세요.
        """
        
        user_prompt = f"""
        다음 음성 인식 결과를 분석하여 메뉴와 수량을 정확히 인식해주세요.
        수량은 반드시 보존하고, 메뉴는 정확한 이름으로 변환해주세요.
        입력: '{text}'
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        # 결과에서 불필요한 설명 제거
        result = response.choices[0].message.content.strip()
        
        # "→" 기호 이후의 텍스트만 추출 (예시 형식이 있을 경우)
        if "→" in result:
            result = result.split("→")[-1].strip()
            
        return result
        
    except Exception as e:
        print(f"Error in GPT correction: {e}")
        return text

def extract_menus_with_quantity(text: str) -> dict:
    """Extract menu items and their quantities from text with improved quantity handling"""
    orders = {}
    
    # Korean number to integer mapping (expanded)
    korean_nums = {
        "한": 1, "하나": 1, "일": 1,
        "두": 2, "둘": 2, "이": 2, "두 개": 2,
        "세": 3, "셋": 3, "삼": 3, "서": 3, "세 개": 3,
        "네": 4, "넷": 4, "사": 4, "세 개": 4,
        "다섯": 5, "오": 5, "다섯 개": 5,
        "여섯": 6, "육": 6, "여섯 개": 6,
        "일곱": 7, "칠": 7, "일곱 개": 7,
        "여덟": 8, "팔": 8, "여덟 개": 8,
        "아홉": 9, "구": 9, "아홉 개": 9,
        "열": 10, "십": 10, "열 개": 10,
        "스물": 20, "이십": 20, "스무": 20,
        "서른": 30, "삼십": 30,
        "마흔": 40, "사십": 40,
        "쉰": 50, "오십": 50
    }
    
    # Menu keywords and their variations
    menu_keywords = {
        "돼지국밥": ["돼지국밥", "돼지 국밥", "돼지"],
        "순대국밥": ["순대국밥", "순대 국밥", "순대"],
        "내장국밥": ["내장국밥", "내장 국밥", "내장"],
        "섞어국밥": ["섞어국밥", "섞어 국밥", "섞어"],
        "수육 반접시": ["수육 반접시", "수육 반 접시", "반접시"],
        "수육 한접시": ["수육 한접시", "수육 한 접시", "수육"]
    }
    
    # Process each menu item
    for menu, keywords in menu_keywords.items():
        for keyword in keywords:
            if keyword in text:
                # Find quantity patterns before and after the keyword
                patterns = [
                    # Patterns for numbers with counters (e.g., "두 개", "세 그릇")
                    rf"([가-힣]+)\s*(?:개|그릇|접시|인분|인승|명|병|잔)\s*{keyword}",
                    rf"{keyword}\s*([가-힣]+)\s*(?:개|그릇|접시|인분|인승|명|병|잔)",
                    # Patterns for numbers (e.g., "두", "세")
                    rf"([가-힣]+)\s+{keyword}",
                    rf"{keyword}\s+([가-힣]+)",
                    # Patterns for numeric digits (e.g., "2개", "3 그릇")
                    rf"(\d+)\s*(?:개|그릇|접시|인분|인승|명|병|잔)?\s*{keyword}",
                    rf"{keyword}\s*(\d+)\s*(?:개|그릇|접시|인분|인승|명|병|잔)?"
                ]
                
                qty = 1  # Default quantity
                
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match and match.group(1):
                        qty_str = match.group(1).strip()
                        
                        # Try to convert to integer
                        try:
                            qty = int(qty_str)
                            break
                        except ValueError:
                            # If not a digit, try Korean number
                            qty = korean_nums.get(qty_str, 1)
                            if qty != 1:  # If found in korean_nums
                                break
                
                # Add to orders
                if menu in orders:
                    orders[menu] += qty
                else:
                    orders[menu] = qty
                
                # Remove the matched text to avoid duplicate processing
                text = text.replace(keyword, "", 1)
                break
    
    return orders

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "menu_list": menu_list})

@app.get("/menu")
async def get_menu():
    """현재 메뉴 목록을 반환합니다."""
    return {"menu_items": menu_items}

@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    try:
        # Read and process audio
        audio_data = await audio.read()
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Convert to text using Whisper
        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        generated_ids = model.generate(input_features)
        recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Correct text using GPT
        corrected_text = correct_text_with_gpt(recognized_text)
        
        # Extract order items
        order_items = extract_menus_with_quantity(corrected_text)
        
        return {
            "original_text": recognized_text,
            "corrected_text": corrected_text,
            "order_items": order_items
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/process-voice")
async def process_voice_command(command: VoiceCommand):
    """
    프론트엔드에서 전송한 음성 명령을 처리합니다.
    """
    try:
        print(f"\n[음성 인식 로그]")
        print(f"원본 텍스트: {command.text}")
        
        # 음성 텍스트를 GPT로 교정
        corrected_text = correct_text_with_gpt(command.text)
        print(f"교정된 텍스트: {corrected_text}")
        
        # 주문 항목 추출
        order_items = extract_menus_with_quantity(corrected_text)
        print(f"추출된 주문 항목: {order_items}")
        
        return {
            "status": "success",
            "original_text": command.text,
            "corrected_text": corrected_text,
            "order_items": order_items
        }
    except Exception as e:
        error_msg = f"음성 처리 중 오류 발생: {str(e)}"
        print(f"[오류] {error_msg}")
        return {
            "status": "error",
            "message": error_msg
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
