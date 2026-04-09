# ingestion/parse.py
import pdfplumber
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from dataclasses import dataclass, field
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    domain: str
    source_file: str
    sections: list[dict] = field(default_factory=list)
    # Each section: {"id": "591-3", "title": "...", "text": "...", "parent": "Part II"}


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common PDF artifacts."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # fix hyphenation
    text = text.strip()
    return text

def _open_section(match, current_article, page_num, doc) -> dict:
    """Save current section (if any) and open a new one."""
    title = match.group(2).strip()
    return {
        "id": match.group(1),
        "title": title,
        "text": "",
        "parent": current_article,
        "page": page_num + 1,
        "_title_complete": _title_looks_complete(title),
    }


def _title_looks_complete(title: str) -> bool:
    """
    A heading title is complete if it ends with sentence-ending punctuation
    or a closing bracket. Incomplete titles get continued on the next line.
    """
    return bool(re.search(r'[.!?\]]$', title.strip()))


# ── Line classifiers ───────────────────────────────────────────────────────────

HEADER_PATTERNS = [
    re.compile(r'^TORONTO MUNICIPAL CODE$'),
    re.compile(r'^CHAPTER \d+,'),
]

# Matches: "547-2 January 1, 2025" or "591-4 March 15, 2024"
FOOTER_PATTERN = re.compile(r'^\d{3}-\d+\s+\w+ \d+, \d{4}$')

# Matches: "ARTICLE 1", "ARTICLE 2", "ARTICLE V" etc.
ARTICLE_PATTERN = re.compile(r'^ARTICLE\s+(\d+|[IVXLC]+)$', re.IGNORECASE)

# Matches real section headings: "§ 547-1.1. Title text"
# The § must be at the start of the line, and title must start with capital letter
SECTION_HEADING = re.compile(r'^§\s+(\d{3}-\d+(?:\.\d+)?)\.\s*([A-Z].*)')


def is_header(line: str) -> bool:
    return any(p.match(line) for p in HEADER_PATTERNS)


def is_footer(line: str) -> bool:
    return bool(FOOTER_PATTERN.match(line))


def is_toc_line(line: str) -> bool:
    """
    ToC lines look identical to real headings — they start with §.
    The difference is that in the ToC, § lines are never followed by prose.
    We handle this at the document level, not the line level.
    """
    return bool(SECTION_HEADING.match(line))


def parse_pdf(pdf_path: Path, domain: str) -> ParsedDocument:
    """
    Parse a by-law PDF into structured sections.
    Detects section boundaries by looking for numbered headings
    like '591-3', 'PART II', 'Article II', etc.
    """
    doc = ParsedDocument(domain=domain, source_file=str(pdf_path))

    current_article = None
    current_section = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw_text = page.extract_text()
            if not raw_text:
                continue

            for line in raw_text.split('\n'):
                line = clean_text(line)
                if not line or is_header(line) or is_footer(line):
                    continue

                article_match = ARTICLE_PATTERN.match(line)
                section_match = SECTION_HEADING.match(line)

                if article_match:
                    current_article = f"ARTICLE {article_match.group(1).upper()}"
                    continue

                if section_match:
                    # Save completed section
                    if current_section:
                        doc.sections.append(current_section)

                    current_section = {
                        "id": section_match.group(1),
                        "title": section_match.group(2).strip(),
                        "text": "",
                        "parent": current_article,
                        "page": page_num + 1,
                    }
                    continue

                # Body text — accumulate into current section
                if current_section is not None:
                    current_section["text"] += " " + line

    # Don't lose the last section
    if current_section:
        doc.sections.append(current_section)

    logger.info(f"  → {len(doc.sections)} sections across "
                f"{len(set(s['parent'] for s in doc.sections))} articles ({domain})")

    return doc

def _infer_parent(section_id: str, existing_sections: list[dict]) -> str | None:
    """
    Infer the parent section from the section ID.
    e.g. '591-3' belongs to the most recent 'PART X' heading.
    """
    for section in reversed(existing_sections):
        if section["id"].upper().startswith("PART"):
            return section["id"]
    return None