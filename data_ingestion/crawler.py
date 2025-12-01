"""
Sodick RAG Data Ingestion Pipeline - Complete Edition
======================================================
One-time data collection system for the Sodick RAG Assistant.
Includes web crawling, PDF processing, and English filtering.

Pipeline Flow:
1. Crawl website → Extract content with metadata
2. Download PDFs → Process with OCR and tables  
3. Combine all data → Detect languages (EN/JA/ZH)
4. Filter English → Keep only English documents
5. Save filtered data → Ready for embedding and Pinecone upload

"""

import os
import re
import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse

# Web scraping
import requests
from bs4 import BeautifulSoup

# PDF processing
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

# OCR and tables
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print(" OCR not available. Install: pip install pytesseract pillow")

try:
    import camelot
    import pandas as pd
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print(" Table extraction not available. Install: pip install camelot-py[cv]")

# Language detection
from langdetect import detect, DetectorFactory

# LangChain compatibility
from langchain_core.documents import Document

# Progress tracking
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
DetectorFactory.seed = 0  # Consistent language detection


# CONFIGURATION

class CrawlerConfig:
    """Configuration for the data ingestion pipeline"""
    
    # Website settings
    BASE_URL = "https://www.sodick.co.jp/en"
    MAX_WEB_PAGES = None  # None = unlimited, int = limit
    CRAWL_DELAY = 0.5  # Seconds between requests
    
    # PDF settings
    PDF_DOWNLOAD_DIR = "./research/data"
    PDF_PROCESSED_DIR = "./research/processed"
    MAX_PDF_PAGES = 2000  # Maximum pages to search for PDFs
    
    # Language settings - ONLY these 3 languages
    SUPPORTED_LANGUAGES = ['english', 'japanese', 'chinese']
    
    # **ENGLISH FILTERING - Set to True to filter English only**
    FILTER_ENGLISH_ONLY = True  
    
    # Output settings
    OUTPUT_ALL_LANGUAGES = "./research/data/sodick_all_languages.pkl"
    OUTPUT_ENGLISH_ONLY = "./research/data/sodick_english_only.pkl"
    
    # OCR settings
    OCR_DPI = 300
    OCR_LANGUAGES = 'eng+jpn+chi_sim'
    
    # User agent
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


# UTILITY FUNCTIONS

def detect_content_language(text: str) -> str:
    """
    Detect language - English, Japanese, Chinese only
    
    Args:
        text: Text content to analyze
        
    Returns:
        str: Language name ('english', 'japanese', 'chinese', 'unknown')
    """
    if not text or len(text.strip()) < 20:
        return 'unknown'
    
    try:
        sample = text[:1000].strip()
        lang_code = detect(sample)
        
        lang_map = {
            'en': 'english',
            'ja': 'japanese',
            'zh-cn': 'chinese',
            'zh-tw': 'chinese'
        }
        
        return lang_map.get(lang_code, 'unknown')
    except:  
        return 'unknown'


def extract_year(source: str, text: str) -> int:
    """
    Extract year from filename/URL or content
    
    Args:
        source: Filename or URL
        text: Content text
        
    Returns:
        int: Extracted year (0 if not found)
    """
    # Try source first
    year_match = re.search(r'20\d{2}|19\d{2}', source)
    if year_match:
        year = int(year_match.group())
        if 1990 <= year <= 2030:
            return year
    
    # Try content
    year_patterns = [
        r'(?:fiscal|financial)\s+year\s+(\d{4})',
        r'(?:copyright|©)\s+(\d{4})',
        r'\b(20\d{2})\b'
    ]
    
    for pattern in year_patterns:
        matches = re.finditer(pattern, text[:5000], re.IGNORECASE)
        for match in matches:
            try:
                year = int(match.group(1))
                if 1990 <= year <= 2030:
                    return year
            except:
                continue
    
    return 0


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem use"""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'
    return filename


# WEB CRAWLING

class WebCrawler:
    """Crawl Sodick website and extract content with metadata"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.visited: Set[str] = set()
        self.base_domain = urlparse(config.BASE_URL).netloc
    
    def crawl(self) -> List[Document]:
        """Main crawling function"""
        print("\n" + "="*70)
        print("WEBSITE CRAWLING")
        print("="*70)
        print(f"Starting URL: {self.config.BASE_URL}")
        print(f"Max pages: {'Unlimited' if self.config.MAX_WEB_PAGES is None else self.config.MAX_WEB_PAGES}")
        print(f"Languages: English, Japanese, Chinese")
        print("="*70 + "\n")
        
        to_visit = [self.config.BASE_URL]
        all_documents = []
        
        if self.config.MAX_WEB_PAGES:
            pbar = tqdm(total=self.config.MAX_WEB_PAGES, desc="Crawling pages")
        else:
            pbar = tqdm(desc="Crawling pages")
        
        while to_visit:
            if self.config.MAX_WEB_PAGES and len(self.visited) >= self.config.MAX_WEB_PAGES:
                break
            
            url = to_visit.pop(0)
            
            if url in self.visited:
                continue
            
            self.visited.add(url)
            docs = self._crawl_single_page(url)
            all_documents.extend(docs)
            
            new_links = self._extract_links(url)
            to_visit.extend(new_links)
            
            pbar.update(1)
            time.sleep(self.config.CRAWL_DELAY)
        
        pbar.close()
        self._print_summary(all_documents)
        
        return all_documents
    
    def _crawl_single_page(self, url: str) -> List[Document]:
        """Crawl single page and return documents"""
        try:
            response = requests.get(
                url,
                timeout=15,
                headers={'User-Agent': self.config.USER_AGENT}
            )
            
            if 'text/html' not in response.headers.get('content-type', ''):
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = self._extract_title(soup)
            description = self._extract_description(soup)
            main_text = self._extract_main_content(soup)
            tables = self._extract_tables(soup)
            image_info = self._extract_image_info(soup)
            
            all_text = f"{title} {description} {main_text}"
            language = detect_content_language(all_text)
            year = extract_year(url, all_text)
            
            documents = []
            
            base_metadata = {
                'source': url,
                'source_type': 'website',
                'title': title,
                'language': language,
                'year': year,
                'timestamp': datetime.now().isoformat(),
                'domain': self.base_domain,
                'has_images': image_info['has_images'],
                'image_count': image_info['image_count'],
                'has_tables': len(tables) > 0,
                'table_count': len(tables)
            }
            
            content_parts = [title, description, main_text]
            page_content = '\n\n'.join(filter(None, content_parts))
            
            main_doc = Document(
                page_content=page_content,
                metadata={
                    **base_metadata,
                    'content_type': 'main_content',
                    'content_length': len(page_content)
                }
            )
            documents.append(main_doc)
            
            for table in tables:
                table_doc = Document(
                    page_content=table['text'],
                    metadata={
                        **base_metadata,
                        'content_type': 'table',
                        'table_id': table['table_id'],
                        'table_shape': f"{table['row_count']}x{table['column_count']}"
                    }
                )
                documents.append(table_doc)
            
            return documents
            
        except Exception as e:
            return []
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        return ''
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta and meta.get('content'):
            return meta['content'].strip()
        return ''
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content"""
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
            tag.decompose()
        
        main = (soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_=re.compile('content|main')))
        
        if main:
            text = main.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract tables from HTML"""
        tables = []
        
        for idx, table in enumerate(soup.find_all('table')):
            try:
                headers = []
                header_row = table.find('tr')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                
                table_data = []
                for row in table.find_all('tr')[1:]:
                    cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    if cells:
                        table_data.append(cells)
                
                if table_data:
                    table_text = self._format_table_to_text(headers, table_data)
                    tables.append({
                        'table_id': idx + 1,
                        'text': table_text,
                        'row_count': len(table_data),
                        'column_count': len(headers) if headers else len(table_data[0])
                    })
            except:
                continue
        
        return tables
    
    def _format_table_to_text(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format table to searchable text"""
        lines = []
        
        if headers:
            lines.append(f"TABLE COLUMNS: {' | '.join(headers)}")
            lines.append("-" * 50)
            
            for row in rows:
                row_parts = []
                for col_idx, value in enumerate(row):
                    if value.strip():
                        col_name = headers[col_idx] if col_idx < len(headers) else f"Col{col_idx}"
                        row_parts.append(f"{col_name}: {value}")
                
                if row_parts:
                    lines.append(" | ".join(row_parts))
        else:
            for row in rows:
                if any(cell.strip() for cell in row):
                    lines.append(" | ".join(cell for cell in row if cell.strip()))
        
        return "\n".join(lines)
    
    def _extract_image_info(self, soup: BeautifulSoup) -> Dict:
        """Extract image information"""
        images = soup.find_all('img')
        return {
            'has_images': len(images) > 0,
            'image_count': len(images)
        }
    
    def _extract_links(self, url: str) -> List[str]:
        """Extract valid links from page"""
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={'User-Agent': self.config.USER_AGENT}
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for a in soup.find_all('a', href=True):
                full_url = urljoin(url, a['href'])
                parsed = urlparse(full_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                if (parsed.netloc == self.base_domain and
                    clean_url not in self.visited and
                    not any(clean_url.endswith(ext) for ext in 
                           ['.pdf', '.jpg', '.png', '.gif', '.zip', '.doc', '.xls'])):
                    links.append(clean_url)
            
            return list(set(links))
            
        except:
            return []
    
    def _print_summary(self, documents: List[Document]):
        """Print crawling summary"""
        print(f"\n{'='*70}")
        print(" WEBSITE CRAWLING COMPLETE")
        print(f"{'='*70}")
        print(f" Pages visited: {len(self.visited)}")
        print(f" Documents created: {len(documents)}")
        
        languages = {}
        for doc in documents:
            if doc.metadata.get('content_type') == 'main_content':
                lang = doc.metadata['language']
                languages[lang] = languages.get(lang, 0) + 1
        
        print(f"\n Languages:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            print(f"   {lang}: {count}")
        
        print(f"{'='*70}\n")


# PDF DISCOVERY & DOWNLOADING

class PDFDownloader:
    """Find and download PDFs from website"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
    
    def find_and_download(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Find PDFs and download them"""
        pdf_links = self._find_pdf_links()
        successful, failed = self._download_pdfs(pdf_links)
        return successful, failed
    
    def _find_pdf_links(self) -> List[str]:
        """Crawl website for PDF links"""

        
        pdf_links = set()
        to_visit = [self.config.BASE_URL]
        visited = set()
        base_domain = urlparse(self.config.BASE_URL).netloc
        
        pbar = tqdm(total=self.config.MAX_PDF_PAGES, desc="Searching pages")
        
        while to_visit and len(visited) < self.config.MAX_PDF_PAGES:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            visited.add(url)
            
            try:
                response = requests.get(
                    url,
                    timeout=10,
                    headers={'User-Agent': self.config.USER_AGENT}
                )
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)
                    
                    if '.pdf' in full_url.lower() and parsed.netloc == base_domain:
                        if full_url not in pdf_links:
                            pdf_links.add(full_url)
                            tqdm.write(f"  Found: {os.path.basename(full_url)}")
                    
                    elif (full_url.startswith(self.config.BASE_URL) and
                          full_url not in visited and
                          full_url not in to_visit):
                        to_visit.append(full_url)
                
            except:
                pass
            
            pbar.update(1)
            time.sleep(self.config.CRAWL_DELAY)
        
        pbar.close()
        
        print(f"\nSearched {len(visited)} pages")
        print(f"Found {len(pdf_links)} PDF files\n")
        
        return list(pdf_links)
    
    def _download_pdfs(self, pdf_links: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Download PDF files"""
        os.makedirs(self.config.PDF_DOWNLOAD_DIR, exist_ok=True)
        
        successful = []
        failed = []
        
        print(f"  Downloading {len(pdf_links)} PDFs...\n")
        
        for i, pdf_url in enumerate(tqdm(pdf_links, desc="Downloading")):
            try:
                response = requests.get(
                    pdf_url,
                    timeout=30,
                    headers={'User-Agent': self.config.USER_AGENT}
                )
                response.raise_for_status()
                
                filename = os.path.basename(urlparse(pdf_url).path)
                filename = sanitize_filename(filename)
                
                if not filename or not filename.endswith('.pdf'):
                    filename = f"document_{i+1}.pdf"
                
                file_path = os.path.join(self.config.PDF_DOWNLOAD_DIR, filename)
                if os.path.exists(file_path):
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{i+1}{ext}"
                    file_path = os.path.join(self.config.PDF_DOWNLOAD_DIR, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                successful.append(file_path)
                time.sleep(0.5)
                
            except Exception as e:
                failed.append((pdf_url, str(e)))
    
        print(f"\n Downloaded: {len(successful)}/{len(pdf_links)}")
        print(f" Failed: {len(failed)}\n")
        
        return successful, failed


# PDF PROCESSING

class PDFProcessor:
    """Process PDFs with OCR and table extraction"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
    
    def process(self, pdf_directory: str) -> List[Document]:
        """Process all PDFs in directory"""
        print("\n" + "="*70)
        print("PROCESSING PDFs")
        print("="*70)

        print("="*70 + "\n")
        
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        
        all_documents = []
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
            docs = self._process_single_pdf(pdf_file)
            all_documents.extend(docs)
        
        self._print_summary(all_documents, len(pdf_files))
        
        return all_documents
    
    def _process_single_pdf(self, pdf_path: Path) -> List[Document]:
        """Process single PDF file"""
        try:
            doc = fitz.open(str(pdf_path))
            
            all_text = ""
            for page in doc:
                all_text += page.get_text() + "\n"
            
            language = detect_content_language(all_text)
            year = extract_year(pdf_path.name, all_text)
            
            print(f"  Pages: {len(doc)}")
            print(f"  Language: {language}")
            print(f"  Year: {year if year > 0 else 'N/A'}")
            
            documents = []
            
            for page_num, page in enumerate(doc):
                page_text = self._extract_text_with_ocr(page)
                
                if not page_text.strip():
                    continue
                
                base_metadata = {
                    'source': f"file:///{pdf_path.name}",
                    'source_type': 'pdf',
                    'filename': pdf_path.name,
                    'page_number': page_num + 1,
                    'total_pages': len(doc),
                    'language': language,
                    'year': year,
                    'timestamp': datetime.now().isoformat(),
                    'domain': 'sodick.co.jp',
                    'has_images': len(page.get_images()) > 0,
                    'image_count': len(page.get_images()),
                    'content_length': len(page_text)
                }
                
                tables = self._extract_tables(pdf_path, page_num + 1)
                
                if tables:
                    print(f"      📊 Found {len(tables)} table(s)")
                    for table in tables:
                        table_doc = Document(
                            page_content=table['text'],
                            metadata={
                                **base_metadata,
                                'content_type': 'table',
                                'table_id': table['table_id']
                            }
                        )
                        documents.append(table_doc)
                
                content_doc = Document(
                    page_content=page_text,
                    metadata={
                        **base_metadata,
                        'content_type': 'main_content',
                        'has_tables': len(tables) > 0
                    }
                )
                documents.append(content_doc)
            
            doc.close()
            print(f"   ✅ Processed {len(doc)} pages")
            
            return documents
            
        except Exception as e:
            print(f"   ERROR: {str(e)[:100]}")
            return []
    
    def _extract_text_with_ocr(self, page) -> str:
        """Extract text with OCR fallback"""
        text = page.get_text()
        
        if OCR_AVAILABLE and len(text.strip()) < 50:
            try:
                pix = page.get_pixmap(dpi=self.config.OCR_DPI)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                ocr_text = pytesseract.image_to_string(
                    img,
                    lang=self.config.OCR_LANGUAGES,
                    config='--psm 6'
                )
                
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            except:
                pass
        
        return text
    
    def _extract_tables(self, pdf_path: Path, page_num: int) -> List[Dict]:
        """Extract tables using Camelot"""
        if not CAMELOT_AVAILABLE:
            return []
        
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_num),
                flavor='lattice',
                suppress_stdout=True
            )
            
            if len(tables) == 0:
                tables = camelot.read_pdf(
                    str(pdf_path),
                    pages=str(page_num),
                    flavor='stream',
                    suppress_stdout=True
                )
            
            extracted = []
            for idx, table in enumerate(tables):
                df = table.df
                df = df.replace('', pd.NA).dropna(how='all').dropna(axis=1, how='all')
                
                if df.shape[0] > 0 and df.shape[1] > 0:
                    table_text = self._format_table_to_text(df)
                    extracted.append({
                        'table_id': idx + 1,
                        'text': table_text
                    })
            
            return extracted
            
        except:
            return []
    
    def _format_table_to_text(self, df: pd.DataFrame) -> str:
        """Format DataFrame to searchable text"""
        if df.empty:
            return ""
        
        rows = []
        headers = [str(col).strip() for col in df.columns if str(col).strip()]
        
        if headers:
            rows.append(f"TABLE COLUMNS: {' | '.join(headers)}")
            rows.append("-" * 50)
        
        for idx, row in df.iterrows():
            row_parts = []
            for col_idx, col in enumerate(df.columns):
                value = str(row[col]).strip()
                if value and value != 'nan':
                    col_name = headers[col_idx] if col_idx < len(headers) else f"Col{col_idx}"
                    row_parts.append(f"{col_name}: {value}")
            
            if row_parts:
                rows.append(" | ".join(row_parts))
        
        return "\n".join(rows)
    
    def _print_summary(self, documents: List[Document], total_files: int):
        """Print processing summary"""
        print(f"\n{'='*70}")
        print("PDF PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Files processed: {total_files}")
        print(f"Documents created: {len(documents)}")
        
        tables = sum(1 for doc in documents if doc.metadata.get('content_type') == 'table')
        print(f"Tables extracted: {tables}")
        
        print(f"{'='*70}\n")


# ENGLISH FILTERING (KEY COMPONENT)

class LanguageFilter:
    """Filter documents by language - critical for English-only RAG"""
    
    @staticmethod
    def filter_english(documents: List[Document]) -> List[Document]:
        """
        Filter to keep only English documents
        
      
        1. BAAI/bge-small-en-v1.5 embeddings are optimized for English
        2. OpenAI GPT performs best with English retrieval
        3. Simplifies user experience
        
        Args:
            documents: All documents with language metadata
            
        Returns:
            List[Document]: English documents only
        """
       
        
        english_docs = []
        
        for doc in documents:
            language = doc.metadata.get('language', '').lower()
            
            # Check metadata first
            if language in ['english', 'en', 'eng']:
                english_docs.append(doc)
            
            # Fallback: Detect from content if metadata missing/unknown
            elif not language or language == 'unknown':
                try:
                    detected = detect(doc.page_content[:1000])
                    if detected == 'en':
                        # Update metadata
                        doc.metadata['language'] = 'english'
                        english_docs.append(doc)
                except:
                    pass  # Skip if detection fails
        
        # Statistics
        print(f" FILTERING RESULTS:")
        print(f"   Total documents: {len(documents)}")
        print(f"   English documents: {len(english_docs)}")
        print(f"   Filtered out: {len(documents) - len(english_docs)}")
        
        # Language breakdown
        languages = {}
        for doc in documents:
            lang = doc.metadata.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"\n Original language distribution:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(documents)) * 100
            print(f"   {lang}: {count} ({percentage:.1f}%)")
        
        # Content type breakdown
        content_types = {}
        for doc in english_docs:
            ct = doc.metadata.get('content_type', 'unknown')
            content_types[ct] = content_types.get(ct, 0) + 1
        
        print(f"\n English documents by type:")
        for ct, count in content_types.items():
            print(f"   {ct}: {count}")
        
        print(f"\n{'='*70}")
        print(f" Filtering complete! Ready for embedding.")
        print(f"{'='*70}\n")
        
        return english_docs


# DATA COMBINER

class DataCombiner:
    """Combine web and PDF data"""
    
    @staticmethod
    def combine(web_docs: List[Document], pdf_docs: List[Document], output_path: str):
        """Combine and save all documents"""
        all_docs = web_docs + pdf_docs
        
        print("SAVING COMBINED DATA")
        print(f"Web documents: {len(web_docs)}")
        print(f"PDF documents: {len(pdf_docs)}")
        print(f"Total documents: {len(all_docs)}")
        print(f"Output: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(all_docs, f)
        
        print(f"Saved successfully!\n")
        
        return all_docs


# MAIN EXECUTION PIPELINE

def run_full_pipeline():
    """
    Execute complete data ingestion pipeline with English filtering
    
    Pipeline Flow:
    1. Crawl website → Get all content (EN/JA/ZH)
    2. Download PDFs → Get all PDFs (EN/JA/ZH)
    3. Process PDFs → Extract text, tables, metadata
    4. Combine data → Merge web + PDF documents
    5. **Filter English** → Keep only English for embedding
    6. Save filtered data → Ready for Pinecone upload
    """

    
    config = CrawlerConfig()
    
    # Step 1: Crawl website
    print(" STEP 1: Web Crawling")
    crawler = WebCrawler(config)
    web_docs = crawler.crawl()
    
    # Step 2: Find and download PDFs
    print("\n STEP 2: PDF Discovery & Download")
    downloader = PDFDownloader(config)
    successful_downloads, failed_downloads = downloader.find_and_download()
    
    # Step 3: Process PDFs
    print("\n STEP 3: PDF Processing")
    processor = PDFProcessor(config)
    pdf_docs = processor.process(config.PDF_DOWNLOAD_DIR)
    
    # Step 4: Combine all data
    print("\n STEP 4: Combining Data")
    combiner = DataCombiner()
    all_docs = combiner.combine(web_docs, pdf_docs, config.OUTPUT_ALL_LANGUAGES)
    
    # Step 5: Filter to English ONLY 
    if config.FILTER_ENGLISH_ONLY:
      
        filter_tool = LanguageFilter()
        english_docs = filter_tool.filter_english(all_docs)
        
        # Save English-only version (goes to Pinecone)
        with open(config.OUTPUT_ENGLISH_ONLY, 'wb') as f:
            pickle.dump(english_docs, f)
        print(f"Saved English-only data to: {config.OUTPUT_ENGLISH_ONLY}\n")
        
        final_docs = english_docs
    else:
        print("\n STEP 5: Skipping English filtering (using all languages)")
        final_docs = all_docs
    
    
    return final_docs


# ENTRY POINT

if __name__ == "__main__":
    """
    Run this script once to collect all data.
    Not needed for the main RAG application.
    The RAG system uses pre-processed English data from Pinecone.
    """

    config = CrawlerConfig()

    
    # Uncomment to run
    #documents = run_full_pipeline()
    
