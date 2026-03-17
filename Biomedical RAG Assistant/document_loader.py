from langchain_community.document_loaders import UnstructuredPDFLoader
from  langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract


# Explicitly set Tesseract path (required on Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_with_unstructured(pdf_path: str):
    print("Loading document using UnstructuredPDFLoader")

    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()

    return docs


def load_with_ocr(pdf_path: str):

    print("Loading document using OCR")

    images = convert_from_path(pdf_path)
    documents = []

    for page_num, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image, lang="eng")

        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_path,
                "page": page_num
            }
        )

        documents.append(doc)

    return documents
