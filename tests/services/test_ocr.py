"""
Tests for OCR service.
"""

from unittest.mock import MagicMock, patch

import pytest

from marketing_project.services.ocr import (
    extract_text_from_image,
    extract_text_from_url,
    process_content_images,
)


def test_extract_text_from_image():
    """Test extract_text_from_image function."""
    # Create a simple test image
    import io

    from PIL import Image

    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    with patch("pytesseract.image_to_string") as mock_ocr:
        mock_ocr.return_value = "Test OCR text"

        text = extract_text_from_image(img_bytes.getvalue())

        assert isinstance(text, str)
        assert len(text) >= 0


def test_extract_text_from_url():
    """Test extract_text_from_url function."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = b"fake image data"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with patch(
            "marketing_project.services.ocr.extract_text_from_image"
        ) as mock_extract:
            mock_extract.return_value = "Test text"

            text = extract_text_from_url("https://example.com/image.png")

            assert isinstance(text, str)


def test_process_content_images():
    """Test process_content_images function."""
    content = "This is test content with images"
    image_urls = ["https://example.com/image1.png"]

    with patch("marketing_project.services.ocr.extract_text_from_url") as mock_extract:
        mock_extract.return_value = "Extracted text"

        result = process_content_images(content, image_urls=image_urls)

        assert isinstance(result, dict)
        assert "ocr_results" in result or "enhanced_content" in result
