"""
PDF Text Extraction Utility
"""
import PyPDF2
from typing import Optional


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from an uploaded PDF file
    
    Args:
        pdf_file: Django FileField or file-like object
        
    Returns:
        Extracted text as string
    """
    try:
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text_content = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        
        # Join all pages
        full_text = "\n".join(text_content)
        
        return full_text
        
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_pdf_file(file_path: str) -> str:
    """
    Extract text from a PDF file given its path
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as string
    """
    try:
        with open(file_path, 'rb') as file:
            return extract_text_from_pdf(file)
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        raise ValueError(f"Failed to read PDF file: {str(e)}")


def validate_pdf(file) -> bool:
    """
    Validate if the uploaded file is a valid PDF
    
    Args:
        file: Django FileField
        
    Returns:
        True if valid PDF, False otherwise
    """
    try:
        # Check file extension
        if not file.name.lower().endswith('.pdf'):
            return False
        
        # Try to read PDF
        pdf_reader = PyPDF2.PdfReader(file)
        return len(pdf_reader.pages) > 0
        
    except Exception:
        return False
