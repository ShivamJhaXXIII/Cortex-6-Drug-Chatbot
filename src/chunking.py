import re


SECTION_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s+"
    r"([A-Z][A-Z0-9 ,&()'/-]{2,})"
    r"\s*$"
)


def split_into_paragraphs(text):
    """
    Split page text into paragraphs/blocks.
    """

    text = re.sub(r"[ \t]+", " ", text)

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    return [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]


KNOWN_SECTIONS = [
    "INDICATIONS AND USAGE",
    "DOSAGE AND ADMINISTRATION",
    "DOSAGE FORMS AND STRENGTHS",
    "CONTRAINDICATIONS",
    "WARNINGS AND PRECAUTIONS",
    "ADVERSE REACTIONS",
    "DRUG INTERACTIONS",
    "USE IN SPECIFIC POPULATIONS",
    "DRUG ABUSE AND DEPENDENCE",
    "OVERDOSAGE",
    "DESCRIPTION",
    "CLINICAL PHARMACOLOGY",
    "NONCLINICAL TOXICOLOGY",
    "CLINICAL STUDIES",
    "HOW SUPPLIED/STORAGE AND HANDLING",
    "PATIENT COUNSELING INFORMATION",
]


def detect_section(text):
    """
    Detect a section only when it appears as a
    heading-like line, not when it is mentioned
    inside normal prose.
    """

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove bullets and common formatting
        clean = re.sub(
            r"^[•▪o\-]+\s*",
            "",
            line
        )

        # Remove section number:
        # "6 ADVERSE REACTIONS"
        # "2.1 DOSAGE..."
        clean = re.sub(
            r"^\d+(?:\.\d+)?\s+",
            "",
            clean
        )

        clean = clean.upper().strip()

        for section in KNOWN_SECTIONS:

            if clean == section:
                return section

    return None

def is_packaging_page(text):

    text_lower = text.lower()

    packaging_terms = [
        "packaging",
        "marketing start",
        "marketing end",
        "ndc:",
        "principal display panel"
    ]

    matches = sum(
        term in text_lower
        for term in packaging_terms
    )

    return matches >= 2

def chunk_pages(
    pages,
    chunk_size=1200
):
    """
    Create paragraph-aware chunks while preserving:

    - document
    - page
    - section
    """

    chunks = []

    for page in pages:
        current_section = "Unknown"
        retrievable = not is_packaging_page(page["text"])
        paragraphs = split_into_paragraphs(
            page["text"]
        )

        current_chunk = ""

        for paragraph in paragraphs:

            detected_section = detect_section(
                paragraph
            )

            if detected_section:
                current_section = detected_section

            # If paragraph is too large,
            # split it separately.
            if len(paragraph) > chunk_size:

                if current_chunk.strip():

                    chunks.append({
                        "text": current_chunk.strip(),
                        "document": page["document"],
                        "page": page["page"],
                        "section": current_section,
                        "retrievable": retrievable
                    })

                    current_chunk = ""

                for start in range(
                    0,
                    len(paragraph),
                    chunk_size
                ):

                    piece = paragraph[
                        start:start + chunk_size
                    ].strip()

                    if piece:

                        chunks.append({
                            "text": piece,
                            "document": page["document"],
                            "page": page["page"],
                            "section": current_section,
                            "retrievable": retrievable
                        })

                continue

            # Normal paragraph
            if (
                len(current_chunk)
                + len(paragraph)
                + 2
                <= chunk_size
            ):

                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += paragraph

            else:

                if current_chunk.strip():

                    chunks.append({
                        "text": current_chunk.strip(),
                        "document": page["document"],
                        "page": page["page"],
                        "section": current_section,
                        "retrievable": retrievable
                    })

                current_chunk = paragraph

        # Remaining chunk
        if current_chunk.strip():

            chunks.append({
                "text": current_chunk.strip(),
                "document": page["document"],
                "page": page["page"],
                "section": current_section,
                "retrievable": retrievable
            })
        

    return chunks