# main.py
import argparse
import logging
import sys
from config import (
    PIPELINE_STEPS,
    INPUT_DIR,
    OUTPUT_DIR,
    CRAWLER_OUTPUT_DIR,
    PDF_DOC_OUTPUT_DIR,
    EMBEDDING_OUTPUT_DIR,
    LLM_PROVIDER,
    OPENAI_API_KEYS,
    MAX_TOKENS,
    VERBOSE
)
from crawler import WebCrawler
from pdf_doc_extractor import PDFExtractor
from embedding_processor import EmbeddingProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline complet")
    parser.add_argument("--run", choices=["all", "crawler", "pdf_doc_extractor", "embedding"], default="all",
                        help="Étape à exécuter. 'all' pour tout le pipeline.")
    parser.add_argument("--resume", action="store_true", help="Reprendre depuis le checkpoint.")
    args = parser.parse_args()

    step = args.run
    resume = args.resume

    logging.basicConfig(level=logging.INFO)

    if step == "all" or step == "crawler":
        crawler = WebCrawler(
            start_url=None,  # Utilise la config
            max_depth=None,
            use_playwright=None,
            download_pdf=None,
            download_doc=None,
            download_image=None,
            download_other=None,
            llm_provider=None,
            api_keys=None,
            max_tokens_per_request=None,
            max_urls=None,
            base_dir=CRAWLER_OUTPUT_DIR
        )
        crawler.crawl()
        if step != "all":
            sys.exit(0)

    if step == "all" or step == "pdf_doc_extractor":
        extractor = PDFExtractor(
            input_dir=CRAWLER_OUTPUT_DIR,  # On récupère les PDF/DOC du dossier du crawler
            output_dir=PDF_DOC_OUTPUT_DIR,
            openai_api_keys=OPENAI_API_KEYS,
            llm_provider=LLM_PROVIDER,
            verbose=VERBOSE
        )
        extractor.process_all_pdfs()
        extractor.process_all_docs()
        if step != "all":
            sys.exit(0)

    if step == "all" or step == "embedding":
        processor = EmbeddingProcessor(
            input_dir=PDF_DOC_OUTPUT_DIR,  # On prend les .txt du dossier pdf_doc_extractor + content_rewritten du crawler si nécessaire
            output_dir=EMBEDDING_OUTPUT_DIR,
            openai_api_keys=OPENAI_API_KEYS,
            llm_provider=LLM_PROVIDER,
            embedding_provider="openai",  # ou autre selon config
            verbose=VERBOSE
        )

        # On veut fusionner (merge) le contenu de pdf_doc_extractor et content_rewritten du crawler
        # On crée un dossier temporaire pour le merging, ou on copie simplement les fichiers txt de pdf_doc_extractor + content_rewritten vers EMBEDDING_OUTPUT_DIR
        # Pour simplifier, on considère qu'on a déjà ce qu'il faut dans PDF_DOC_OUTPUT_DIR (ou faire le merging si besoin).
        # Vous pouvez ajouter du code ici pour fusionner les fichiers du content_rewritten (CRAWLER_OUTPUT_DIR/content_rewritten) et PDF_DOC_OUTPUT_DIR
        # dans un seul endroit avant de lancer l'embedding.
        # Par exemple:
        from shutil import copyfile
        content_rewritten_dir = CRAWLER_OUTPUT_DIR + "/content_rewritten"
        if Path(content_rewritten_dir).exists():
            for f in Path(content_rewritten_dir).glob("*.txt"):
                # Copier dans PDF_DOC_OUTPUT_DIR
                dest = Path(PDF_DOC_OUTPUT_DIR) / f.name
                if not dest.exists():
                    copyfile(f, dest)

        processor.process_all_files()
        sys.exit(0)
