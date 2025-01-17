from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from fastapi import FastAPI, Body
import uvicorn
import requests
import json
from typing import Annotated
app = FastAPI()
load_dotenv()

class ImageToText:

    def __init__(self,image_path):
        self.image_path = image_path

    def extract_text_from_image(self):
        """
        Extracts text from the image using the PaddleOCR library.
        This function:
        - Initializes the OCR engine with custom parameters.
        - Processes the image for resizing if needed.
        - Runs the OCR engine twice with different parameters for better accuracy.
        - Filters, deduplicates, and organizes the OCR results.
        - Groups the detected text into rows based on their vertical alignment.
        - Returns the extracted text in a structured format (as rows of text).
        """
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_limit_side_len=2880,
            det_db_thresh=0.3,
            det_db_box_thresh=0.3,
            rec_batch_num=6,
            drop_score=0.5,
        )
        try:
            ## Resize the image
            self.processed_image = self.process_image()
            self.results = []

            ## Run OCR on the processed image
            self.result = self.ocr.ocr(self.processed_image, cls=True)

            # Check and extend the results from the first OCR run
            if self.result and self.result[0]:
                self.results.extend(self.result[0])

            # Adjust OCR parameters for better detection and rerun
            self.ocr.det_limit_side_len = 3500
            self.result2 = self.ocr.ocr(self.processed_image, cls=True)
            if self.result2 and self.result2[0]:
                self.results.extend(self.result2[0])

            # Deduplicate results based on unique text and position
            self.unique_results = []
            self.seen_texts = set()

            for self.item in self.results:
                if self.item is None:
                    continue
                self.coords, (self.text, self.conf) = self.item
                self.text = self.text.strip()
                self.center_x = sum(self.coord[0] for self.coord in self.coords) / 4
                self.center_y = sum(self.coord[1] for self.coord in self.coords) / 4
                self.pos_id = f"{self.text}_{int(self.center_x)}_{int(self.center_y)}"

                # Only include unique text with confidence > 0.5
                if self.pos_id not in self.seen_texts and self.conf > 0.5:
                    self.seen_texts.add(self.pos_id)
                    self.unique_results.append(self.item)

            # Sort results top-to-bottom, left-to-right
            self.sorted_results = sorted(
                self.unique_results,
                key=lambda x: (
                    sum(self.coord[1] for self.coord in x[0]) / 4,
                    sum(self.coord[0] for self.coord in x[0]) / 4,
                ),
            )
            # Group text into rows based on vertical alignment
            self.extracted_rows = []
            self.current_row = []
            self.last_y = None
            
            # Define how close vertically two lines should be to group them
            self.y_threshold = 10
            for self.item in self.sorted_results:
                self.coords, (self.text, self.conf) = self.item
                self.current_y = sum(self.coord[1] for self.coord in self.coords) / 4

                if self.last_y is None:
                    self.current_row.append(self.text)
                elif abs(self.current_y - self.last_y) > self.y_threshold:
                    if self.current_row:
                        self.extracted_rows.append(self.current_row)
                    self.current_row = [self.text]
                else:
                    self.current_row.append(self.text)

                self.last_y = self.current_y

            if self.current_row:
                self.extracted_rows.append(self.current_row)

            return self.extracted_rows

        except Exception as e:
            return(f"Error processing image: {e}")
            return []

    # for resizing the image
    def process_image(self):
        """
        Prepares the image for OCR by:
        - Reading the image from the given path.
        - Resizing the image to a manageable size while maintaining its aspect ratio.
        - Ensuring the dimensions are within an acceptable range for the OCR engine.
        Returns the processed image.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Could not read image")
        self.height, self.width = self.image.shape[:2]
        self.max_dimension = max(self.height, self.width)
        if self.max_dimension > 4000:
            self.scale_factor = 0.5
        elif self.max_dimension < 1000:
            self.scale_factor = 2.0
        else:
            self.scale_factor = 1.0
        if self.scale_factor != 1.0:
            self.new_width = int(self.width * self.scale_factor)
            self.new_height = int(self.height * self.scale_factor)
            self.image = cv2.resize(self.image, (self.new_width, self.new_height))
        return self.image

@app.post("/locationOfImage")
async def getImagePath(request:Annotated[dict,Body()]=None):
    """
    Endpoint to receive the image path, process the image using the `ImageToText` class,
    extract text, and generate a detailed response using a generative AI model.
    """
    image_path = request.get("image_path")
    callClass = ImageToText(image_path)
    text_data = callClass.extract_text_from_image()
    output=[]
    for row in text_data:
        row_output = " ".join(row)
        output.append(row_output)
    api_key = os.getenv("api_key")
    if api_key is None:
        raise ValueError("Please provide a valid Google API key")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.2,
    )

    prompt_action = f"""
            Based on the product description below, please provide the quantity of sugar:
            Product description: {output}
        """

    response = llm.invoke(prompt_action)
    output_final = response.content.strip()
    return (output_final)


if __name__ == "__main__":
    uvicorn.run(app,host=os.getenv("host"),port=int(os.getenv("port")))
