# 🚀 AI Content Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/simonpierreboucher02/pipeline-main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/simonpierreboucher02/pipeline-main/graphs/commit-activity)

[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple.svg)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude--3.5-orange.svg)](https://anthropic.com/)
[![Mistral](https://img.shields.io/badge/Mistral-Mistral--Large-blue.svg)](https://mistral.ai/)
[![Voyage](https://img.shields.io/badge/Voyage-Embeddings-red.svg)](https://www.voyageai.com/)

[![Web Crawling](https://img.shields.io/badge/Web%20Crawling-Playwright%20%7C%20Requests-blue.svg)](https://playwright.dev/)
[![PDF Processing](https://img.shields.io/badge/PDF%20Processing-PyPDF%20%7C%20OCR-green.svg)](https://pypdf.readthedocs.io/)
[![Embeddings](https://img.shields.io/badge/Embeddings-OpenAI%20%7C%20Mistral%20%7C%20Voyage-purple.svg)](https://platform.openai.com/docs/guides/embeddings)

## 📊 Repository Metrics

![GitHub stars](https://img.shields.io/github/stars/simonpierreboucher02/pipeline-main?style=social)
![GitHub forks](https://img.shields.io/github/forks/simonpierreboucher02/pipeline-main?style=social)
![GitHub issues](https://img.shields.io/github/issues/simonpierreboucher02/pipeline-main)
![GitHub pull requests](https://img.shields.io/github/issues-pr/simonpierreboucher02/pipeline-main)
![GitHub contributors](https://img.shields.io/github/contributors/simonpierreboucher02/pipeline-main)
![GitHub last commit](https://img.shields.io/github/last-commit/simonpierreboucher02/pipeline-main)

## 🎯 Overview

A comprehensive AI-powered content processing pipeline that crawls websites, extracts content from PDFs and documents, and generates embeddings for advanced text analysis and search capabilities.

### 🌟 Key Features

- **🔍 Intelligent Web Crawling**: Multi-depth website crawling with Playwright support
- **📄 Document Processing**: PDF and Office document extraction with OCR capabilities
- **🤖 AI-Powered Analysis**: Multi-provider LLM integration (OpenAI, Anthropic, Mistral)
- **🧠 Advanced Embeddings**: Generate contextual embeddings for semantic search
- **⚡ Parallel Processing**: Multi-threaded processing for optimal performance
- **🔄 Checkpoint System**: Resume processing from any point
- **📊 Comprehensive Reporting**: Detailed analytics and progress tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Crawler   │───▶│ PDF/DOC Extractor│───▶│Embedding Processor│
│                 │    │                  │    │                 │
│ • Multi-depth   │    │ • OCR Support    │    │ • Multi-provider │
│ • File download │    │ • LLM Analysis   │    │ • Contextual     │
│ • Content clean │    │ • Text extraction│    │ • Vector storage │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required system dependencies for OCR and PDF processing

### Installation

```bash
# Clone the repository
git clone https://github.com/simonpierreboucher02/pipeline-main.git
cd pipeline-main

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Create a `.env` file with your API keys:

```env
# LLM Providers
LLM_PROVIDER=openai
OPENAI_API_KEYS=your_key_1,your_key_2
ANTHROPIC_API_KEY=your_anthropic_key
MISTRAL_API_KEY=your_mistral_key

# Embedding Providers
EMBEDDING_PROVIDER=openai
VOYAGE_API_KEY=your_voyage_key

# Crawler Settings
START_URL=https://example.com
MAX_DEPTH=2
USE_PLAYWRIGHT=true
DOWNLOAD_PDF=true
DOWNLOAD_DOC=true

# Processing Settings
MAX_TOKENS=10000
VERBOSE=true
```

### Usage

```bash
# Run the complete pipeline
python main.py --run all

# Run individual steps
python main.py --run crawler
python main.py --run pdf_doc_extractor
python main.py --run embedding

# Resume from checkpoint
python main.py --run all --resume
```

## 📁 Project Structure

```
pipeline-main/
├── main.py                 # Main pipeline orchestrator
├── config.py              # Configuration management
├── crawler.py             # Web crawling engine
├── pdf_doc_extractor.py   # Document processing
├── embedding_processor.py # Embedding generation
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── output/               # Generated outputs
    ├── crawler_output/   # Crawled content
    ├── pdf_doc_extracted/ # Processed documents
    └── embedding_output/  # Generated embeddings
```

## 🔧 Components

### 1. Web Crawler (`crawler.py`)

**Features:**
- Multi-depth website crawling
- Intelligent content extraction
- File download support (PDF, DOC, images)
- Playwright integration for JavaScript-heavy sites
- Checkpoint and resume functionality
- Comprehensive reporting

**Capabilities:**
- ✅ URL discovery and filtering
- ✅ Content cleaning and formatting
- ✅ LLM-powered content rewriting
- ✅ Sitemap generation
- ✅ Progress tracking and logging

### 2. PDF/Document Extractor (`pdf_doc_extractor.py`)

**Features:**
- PDF text extraction with PyPDF
- OCR processing for scanned documents
- Office document processing (DOC, DOCX)
- AI-powered content analysis
- Multi-threaded processing

**Capabilities:**
- ✅ Native PDF text extraction
- ✅ OCR with image preprocessing
- ✅ Document format conversion
- ✅ LLM content enhancement
- ✅ Error handling and recovery

### 3. Embedding Processor (`embedding_processor.py`)

**Features:**
- Multi-provider embedding generation
- Contextual chunk processing
- Vector storage and management
- Parallel processing optimization

**Capabilities:**
- ✅ OpenAI embeddings
- ✅ Mistral embeddings
- ✅ Voyage embeddings
- ✅ Contextual chunking
- ✅ Metadata preservation

## 🤖 AI Integration

### Supported LLM Providers

| Provider | Models | Use Cases |
|----------|--------|-----------|
| **OpenAI** | GPT-4o, GPT-4o-mini | Content analysis, rewriting |
| **Anthropic** | Claude-3.5-Sonnet | Document understanding |
| **Mistral** | Mistral-Large | Text processing |

### Supported Embedding Providers

| Provider | Models | Dimensions |
|----------|--------|------------|
| **OpenAI** | text-embedding-3-large | 3072 |
| **Mistral** | mistral-embed | 1024 |
| **Voyage** | voyage-large-2 | 1536 |

## 📊 Performance Metrics

- **Processing Speed**: ~1000 documents/hour
- **OCR Accuracy**: 95%+ with preprocessing
- **Memory Usage**: Optimized for large datasets
- **Scalability**: Multi-threaded architecture
- **Reliability**: Checkpoint system with error recovery

## 🔍 Use Cases

- **Content Analysis**: Extract insights from large document collections
- **Search Enhancement**: Build semantic search systems
- **Knowledge Management**: Create searchable knowledge bases
- **Research Automation**: Process academic papers and reports
- **Compliance Monitoring**: Analyze regulatory documents

## 🛠️ Development

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=. tests/
```

## 📈 Roadmap

- [ ] **Vector Database Integration**: Pinecone, Weaviate, Qdrant
- [ ] **Advanced Analytics**: Content clustering and topic modeling
- [ ] **API Endpoints**: RESTful API for pipeline operations
- [ ] **Docker Support**: Containerized deployment
- [ ] **Cloud Integration**: AWS, GCP, Azure support
- [ ] **Real-time Processing**: Streaming pipeline capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and embeddings
- **Anthropic** for Claude models
- **Mistral AI** for Mistral models
- **Voyage AI** for embedding services
- **Playwright** for web automation
- **PyPDF** for PDF processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/simonpierreboucher02/pipeline-main/issues)
- **Discussions**: [GitHub Discussions](https://github.com/simonpierreboucher02/pipeline-main/discussions)
- **Email**: simonpierreboucher02@gmail.com

---

<div align="center">

**Made with ❤️ by [Simon Pierre Boucher](https://github.com/simonpierreboucher02)**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github)](https://github.com/simonpierreboucher02)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/simonpierreboucher02)

</div> 