from pathlib import Path 
# pyrefly: ignore [missing-import]
import fitz


def extract_pdf(pdf_path: str)-> list[dict]:
    pdf_path=Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError("PDF does not exist")
    
    doc=fitz.open(pdf_path)
    pages=[]


    for page_number, page in enumerate(doc,start=1):
        text=page.get_text("text")

        text=text.strip()
        if text:
            pages.append({
                "page_number":page_number,
                "text":text
            })
    doc.close()
    return pages    