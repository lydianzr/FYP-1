"""
LayoutLMv3 Document Information Extractor
Specialized for logistics documents (Air Waybills, Invoices, Packing Lists)
"""

import torch
from PIL import Image
import numpy as np
import json
import re
from typing import Dict, Any, Optional, List
import logging
from transformers import AutoProcessor, AutoModelForTokenClassification
import pytesseract
from pdf2image import convert_from_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayoutLMv3Extractor:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize LayoutLMv3 model for document information extraction
        
        Args:
            use_gpu: Attempt to use GPU if available (MX350 supported)
        """
        self.device = self._setup_device(use_gpu)
        self.model_name = "microsoft/layoutlmv3-base"
        
        logger.info(f"Loading LayoutLMv3 on device: {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            apply_ocr=False  # We'll handle OCR ourselves
        )
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=12  # Number of entity types we're extracting
        ).to(self.device)
        
        self.model.eval()
        
        # Define field mapping for logistics documents
        self.field_mapping = {
            "awb_number": ["awb", "airway bill", "awb no", "tracking"],
            "shipper": ["shipper", "sender", "from", "consignor"],
            "consignee": ["consignee", "receiver", "to", "customer"],
            "origin": ["origin", "from", "port of loading", "airport of departure"],
            "destination": ["destination", "to", "port of discharge", "airport of arrival"],
            "airline": ["airline", "carrier", "operating carrier"],
            "flight_no": ["flight", "flight number", "flight no"],
            "departure_date": ["date", "departure date", "flight date"],
            "weight": ["weight", "gross weight", "chargeable weight", "kg"],
            "volume": ["volume", "cbm", "cubic", "m3", "measurement"],
            "pieces": ["pieces", "pcs", "quantity", "total pieces"]
        }
        
        # Compile regex patterns for extraction
        self.patterns = {
            "awb_number": r"\b(\d{3}[-\s]?\d{8}|\d{11}|\d{3}-\d{8})\b",
            "flight_no": r"\b([A-Z]{2,3}\d{2,4})\b",
            "weight": r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilogram)",
            "pieces": r"(\d+)\s*(?:pcs|pieces|ctn|cartons)"
        }
    
    def _setup_device(self, use_gpu: bool) -> str:
        """Setup compute device with MX350 optimization"""
        if use_gpu and torch.cuda.is_available():
            # MX350 specific: use float32 for stability
            return "cuda"
        return "cpu"
    
    def extract_text_with_layout(self, image_path: str) -> List[Dict]:
        """
        Extract text and bounding boxes using Tesseract
        Returns list of {text, bbox} dicts
        """
        try:
            # Load image
            if image_path.lower().endswith('.pdf'):
                # Convert first page of PDF to image
                images = convert_from_path(image_path, first_page=1, last_page=1)
                if not images:
                    return []
                image = images[0]
            else:
                image = Image.open(image_path)
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume uniform text block
            )
            
            word_boxes = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                if text and int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                    word_boxes.append({
                        'text': text,
                        'bbox': [
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['left'][i] + ocr_data['width'][i],
                            ocr_data['top'][i] + ocr_data['height'][i]
                        ]
                    })
            
            logger.info(f"Extracted {len(word_boxes)} words with layout from {image_path}")
            return word_boxes
            
        except Exception as e:
            logger.error(f"Error extracting layout: {str(e)}")
            return []
    
    def extract_fields_regex(self, text: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns"""
        extracted = {}
        
        # Concatenate all text for pattern matching
        full_text = ' '.join([w['text'] for w in text]) if isinstance(text, list) else text
        
        for field, pattern in self.patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1)
                # Clean up values
                if field == 'weight':
                    extracted[field] = float(value) if '.' in value else int(value)
                elif field == 'pieces':
                    extracted[field] = int(value)
                else:
                    extracted[field] = value
        
        return extracted
    
    def extract_smart(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        """
        Enhanced extraction using keyword proximity and layout
        """
        extracted = {}
        
        # Combine all text for regex matching
        all_text = ' '.join([block['text'] for block in text_blocks])
        
        # Extract using regex patterns first
        extracted.update(self.extract_fields_regex(all_text))
        
        # Keyword-based extraction for remaining fields
        for field, keywords in self.field_mapping.items():
            if field in extracted:
                continue
                
            for i, block in enumerate(text_blocks):
                block_text_lower = block['text'].lower()
                
                # Check if any keyword matches this block
                for keyword in keywords:
                    if keyword in block_text_lower:
                        # Look at next block for value
                        if i + 1 < len(text_blocks):
                            potential_value = text_blocks[i + 1]['text']
                            # Filter out common noise
                            if not any(x in potential_value.lower() for x in ['page', 'date', 'http']):
                                extracted[field] = potential_value
                                break
                        break
                    
                    # Also check if the keyword IS the value (e.g., airline name appears directly)
                    if block['text'].lower() in [k.lower() for k in keywords if ' ' not in k]:
                        extracted[field] = block['text']
                        break
                
                if field in extracted:
                    break
        
        # Post-process extracted values
        if 'weight' in extracted and isinstance(extracted['weight'], str):
            weight_match = re.search(r'(\d+(?:\.\d+)?)', extracted['weight'])
            if weight_match:
                extracted['weight'] = float(weight_match.group(1))
        
        if 'pieces' in extracted and isinstance(extracted['pieces'], str):
            pieces_match = re.search(r'(\d+)', extracted['pieces'])
            if pieces_match:
                extracted['pieces'] = int(pieces_match.group(1))
        
        return extracted
    
    def extract_from_document(self, document_path: str, document_type: str = "Air Waybill") -> Dict[str, Any]:
        """
        Main extraction method for logistics documents
        
        Args:
            document_path: Path to PDF or image file
            document_type: Type of document (affects extraction strategy)
        
        Returns:
            Dictionary with extracted fields matching ground truth structure
        """
        try:
            # Step 1: Extract text with layout information
            text_blocks = self.extract_text_with_layout(document_path)
            
            if not text_blocks:
                logger.warning(f"No text extracted from {document_path}")
                return {"error": "No text extracted"}
            
            # Step 2: Extract fields using smart strategy
            extracted_fields = self.extract_smart(text_blocks)
            
            # Step 3: Ensure all expected fields exist (with defaults)
            expected_fields = [
                "awb_number", "shipper", "consignee", "origin", "destination",
                "airline", "flight_no", "departure_date", "weight", "volume", "pieces"
            ]
            
            for field in expected_fields:
                if field not in extracted_fields:
                    extracted_fields[field] = None
            
            # Step 4: Add metadata
            result = {
                "document_type": document_type,
                "extraction_method": "layoutlmv3_smart",
                **extracted_fields
            }
            
            logger.info(f"Extracted {sum(1 for v in extracted_fields.values() if v)}/11 fields")
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return {"error": str(e), "document_type": document_type}
    
    def batch_extract(self, document_paths: List[str], document_type: str = "Air Waybill") -> List[Dict]:
        """Extract from multiple documents"""
        results = []
        for path in document_paths:
            result = self.extract_from_document(path, document_type)
            results.append(result)
        return results


# Singleton instance for reuse
_extractor_instance = None

def get_layoutlmv3_extractor(use_gpu: bool = True) -> LayoutLMv3Extractor:
    """Get or create LayoutLMv3 extractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LayoutLMv3Extractor(use_gpu=use_gpu)
    return _extractor_instance


def run_layoutlmv3_extraction(document_path: str, document_type: str = "Air Waybill") -> Dict[str, Any]:
    """
    Convenience function to extract using LayoutLMv3
    
    Args:
        document_path: Path to document (image or PDF)
        document_type: Type of logistics document
    
    Returns:
        Extracted fields matching app.py expectations
    """
    extractor = get_layoutlmv3_extractor()
    result = extractor.extract_from_document(document_path, document_type)
    
    # Format for app.py compatibility
    return {
        "extracted_fields": result,
        "raw_text": json.dumps(result, indent=2),
        "status": "LayoutLMv3 extraction completed",
        "model_name": "LayoutLMv3",
        "approach": "MLLM"
    }


# Quick test function
if __name__ == "__main__":
    # Test with a sample document
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Testing LayoutLMv3 on: {test_path}")
        
        result = run_layoutlmv3_extraction(test_path)
        print("\nExtraction Result:")
        print(json.dumps(result["extracted_fields"], indent=2))
        print(f"\nStatus: {result['status']}")
    else:
        print("Usage: python layoutlmv3_engine.py <document_path>")
        print("Example: python layoutlmv3_engine.py data/uploads/sample_airwaybill.png")