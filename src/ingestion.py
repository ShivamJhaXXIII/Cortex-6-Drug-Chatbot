import fitz
import os
from pathlib import Path
import re
import easyocr


ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False
)

def extract_pdf_pages(pdf_paths):
    """
    Extract normal PDF text and text from embedded images
    page by page from one or more PDFs.
    """

    pages = []

    if isinstance(pdf_paths, (str, os.PathLike)):
        pdf_paths = [pdf_paths]

    for pdf_path in pdf_paths:

        pdf_path = Path(pdf_path)

        document = fitz.open(pdf_path)

        try:

            for page_number in range(
                document.page_count
            ):

                page = document.load_page(
                    page_number
                )

                # -------------------------
                # Normal PDF text
                # -------------------------

                text = str(
                    page.get_text(
                        "text",
                        sort=True
                    )
                ).strip()

                # -------------------------
                # OCR embedded images
                # -------------------------

                images = page.get_images(
                    full=True
                )

                ocr_text = []

                for image in images:

                    xref = image[0]
                    width = image[2]
                    height = image[3]

                    if width < 100 or height < 100:
                        continue
                    
                    image_data = (
                        document.extract_image(
                            xref
                        )
                    )

                    image_bytes = (
                        image_data["image"]
                    )

                    results = ocr_reader.readtext(
                        image_bytes
                    )

                    cleaned_text = (
                        clean_ocr_results(
                            results
                        )
                    )

                    if cleaned_text:
                        ocr_text.append(
                            cleaned_text
                        )

                # -------------------------
                # Combine page content
                # -------------------------

                page_parts = []

                if text:
                    page_parts.append(text)

                if ocr_text:
                    page_parts.append(
                        "[IMAGE TEXT]\n"
                        + "\n\n".join(
                            ocr_text
                        )
                    )

                final_text = "\n\n".join(
                    page_parts
                )

                # Include pages containing
                # either normal text OR OCR text
                if final_text:

                    pages.append(
                        {
                            "text": final_text,
                            "page": page_number + 1,
                            "document": pdf_path.name,
                        }
                    )

        finally:
            document.close()

    return pages

def is_meaningful_text(text):

    text = text.strip()

    if not text:
        return False

    # At least one alphabetic character
    if not re.search(r"[A-Za-z]", text):
        return False

    # Reject strings that are mostly symbols
    alphanumeric = sum(
        c.isalnum()
        for c in text
    )

    if alphanumeric / max(len(text), 1) < 0.5:
        return False

    return True

def clean_ocr_results(results):

    lines = []

    for bbox, text, confidence in results:

        if confidence < 0.60:
            continue

        if not is_meaningful_text(text):
            continue

        lines.append(text.strip())

    return "\n".join(lines)