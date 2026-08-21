import os
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Drug Information Assistant; Hackathon)"
    )
}

def fetch_official_source(
    url,
    storage_dir="storage"
):
    """
    Resolve an official drug-document URL
    and return a local PDF path.
    """

    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError(
            "Only HTTPS URLs are supported."
        )

    # -------------------------
    # DailyMed
    # -------------------------

    if "dailymed.nlm.nih.gov" in parsed.netloc:

        setid = extract_dailymed_setid(url)

        if setid:

            return download_dailymed(
                url,
                storage_dir
            )

    # -------------------------
    # Direct PDF
    # -------------------------

    if is_pdf_url(url):

        return download_pdf(
            url,
            storage_dir
        )

    # -------------------------
    # Generic official webpage
    # -------------------------

    pdf_links = find_pdf_links(url)

    if not pdf_links:

        raise ValueError(
            "No PDF document was found "
            "on this page."
        )

    # For MVP, use the first PDF.
    return download_pdf(
        pdf_links[0],
        storage_dir
    )

def is_pdf_url(url):
    return url.lower().split("?")[0].endswith(".pdf")


def get_filename_from_url(url):
    path = urlparse(url).path

    filename = os.path.basename(path)

    if not filename:
        filename = "downloaded_document.pdf"

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    return filename


def download_pdf(url, storage_dir="storage"):
    """
    Download a PDF from an official source.
    Returns the local file path.
    """

    os.makedirs(
        storage_dir,
        exist_ok=True
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if (
        "pdf" not in content_type
        and not is_pdf_url(url)
    ):
        raise ValueError(
            "The URL did not return a PDF."
        )

    filename = get_filename_from_url(url)

    filepath = os.path.join(
        storage_dir,
        filename
    )

    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath


def find_pdf_links(page_url):
    """
    Find PDF links on an official webpage.
    """

    response = requests.get(
        page_url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    pdf_links = []

    for link in soup.find_all("a", href=True):

        href = link["href"]

        absolute_url = urljoin(
            page_url,
            href
        )

        if is_pdf_url(absolute_url):

            if absolute_url not in pdf_links:
                pdf_links.append(
                    absolute_url
                )

    return pdf_links

def extract_dailymed_setid(url):
    """
    Extract SET ID from a DailyMed drugInfo URL.

    Example:
    https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=XXXXX
    """

    match = re.search(
        r"setid=([a-fA-F0-9-]+)",
        url
    )

    if match:
        return match.group(1)

    return None

def get_dailymed_pdf_url(setid):
    return (
        "https://dailymed.nlm.nih.gov/"
        "dailymed/downloadpdffile.cfm"
        f"?setId={setid}"
    )

def download_dailymed(url, storage_dir="storage"):
    """
    Download the official DailyMed label PDF.
    """

    setid = extract_dailymed_setid(url)

    if not setid:
        raise ValueError(
            "Could not find DailyMed SET ID in URL."
        )

    pdf_url = get_dailymed_pdf_url(
        setid
    )

    os.makedirs(
        storage_dir,
        exist_ok=True
    )

    response = requests.get(
        pdf_url,
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()

    filename = f"dailymed_{setid}.pdf"

    filepath = os.path.join(
        storage_dir,
        filename
    )

    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath