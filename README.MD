# ScriptumAI

[English] | [Français] | [Español]

## English

### Introduction
ScriptumAI is a private and advanced Retrieval-Augmented Generation platform designed for document ingestion, semantic search, and query processing. It leverages cutting-edge machine learning and natural language processing techniques.

### Key Technologies
- **Backend**: Flask
- **Frontend**: Streamlit
- **ML & NLP**: PyTorch, Ollama API
- **Database**: Chroma
- **Testing**: pytest

### Architecture
The project follows a modular architecture with distinct components for embedding, ingestion, query processing, and retrieval.s

### Key Features

1. **Document Ingestion**
   - Multi-format support (text, PDF, DOCX, HTML, Markdown)
   - Chunking for efficient processing
   - Storage in Chroma for fast retrieval

2. **Query Processing**
   - Uses LLM to generate relevant responses
   - Context-based retrieval using cosine similarity

3. **User Interface**
   - Components for ingestion, queries, and search


4. **Unit Testing**
   - Comprehensive tests for all major components

### Quick Start

#### Prerequisites
Python 3.8+ (tested with 3.11), Flask, Streamlit, Ollama API, Chroma, pytest

#### Installation and Launch
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
5. Launch frontend in a terminal: `streamlit run app.py`

### License

This project is available under a dual license:

1. GNU Affero General Public License v3.0 (AGPL-3.0)
2. Commercial License

For more details, see the [LICENSE](LICENSE) file and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

### Contributing

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Français

### Introduction
ScriptumAI est une plateforme privée et avancée de Génération Augmentée par Récupération conçue pour l'ingestion de documents, la recherche sémantique et le traitement des requêtes. Elle utilise des techniques de pointe en apprentissage automatique et en traitement du langage naturel.

### Technologies Clés
- **Backend**: Flask
- **Frontend**: Streamlit
- **ML & NLP**: API Ollama
- **Base de données**: Chroma
- **Tests**: pytest

### Architecture
Le projet suit une architecture modulaire avec des composants distincts pour l'intégration, l'ingestion, le traitement des requêtes et la récupération.

### Fonctionnalités Principales

1. **Ingestion de Documents**
   - Support multi-format (texte, PDF, DOCX, HTML, Markdown)
   - Découpage pour un traitement efficace
   - Stockage dans Chroma pour une récupération rapide

2. **Traitement des Requêtes**
   - Utilise LLM pour générer des réponses pertinentes
   - Récupération basée sur le contexte utilisant la similarité cosinus

3. **Interface Utilisateur**
   - Composants pour l'ingestion, les requêtes et la recherche
   - Tableau de bord pour les statistiques et l'activité récente

4. **Tests Unitaires**
   - Tests complets pour tous les composants majeurs

### Démarrage Rapide

#### Prérequis
Python 3.8+ (testé su 3.11), Flask, Streamlit, API Ollama, Chroma, pytest

#### Installation et Lancement
1. Cloner le dépôt
2. Créer un environnement virtuel
3. Installer les dépendances : `pip install -r requirements.txt`
4. Dans un terminal, lancer le frontend : `streamlit run app.py`

### Licence

Ce projet est disponible sous une double licence :

1. Licence publique générale GNU Affero v3.0 (AGPL-3.0)
2. Licence commerciale

Pour plus de détails, consultez le fichier [LICENSE](LICENSE) et [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

### Contribuer

Veuillez lire notre [CONTRIBUTING.md](CONTRIBUTING.md) pour plus de détails sur notre code de conduite et le processus de soumission des pull requests.

## Español

### Introducción
ScriptumAI es una plataforma privado y avanzada de Generación Aumentada por Recuperación diseñada para la ingestión de documentos, búsqueda semántica y procesamiento de consultas. Utiliza técnicas de vanguardia en aprendizaje automático y procesamiento del lenguaje natural.

### Tecnologías Clave
- **Frontend**: Streamlit
- **ML & NLP**:  API de Ollama
- **Base de datos**: Chroma
- **Pruebas**: pytest

### Arquitectura
El proyecto sigue una arquitectura modular con componentes distintos para la incrustación, ingestión, procesamiento de consultas y recuperación.

### Características Principales

1. **Ingestión de Documentos**
   - Soporte multi-formato (texto, PDF, DOCX, HTML, Markdown)
   - Fragmentación para procesamiento eficiente
   - Almacenamiento en Chroma para recuperación rápida

2. **Procesamiento de Consultas**
   - Usa LLM para generar respuestas relevantes
   - Recuperación basada en contexto usando similitud del coseno

3. **Interfaz de Usuario**
   - Componentes para ingestión, consultas y búsqueda
   - Panel de control para estadísticas y actividad reciente

4. **Pruebas Unitarias**
   - Pruebas exhaustivas para todos los componentes principales

### Inicio Rápido

#### Requisitos Previos
Python 3.8+ (probado en 3.11), Flask, Streamlit, API de Ollama, Chroma, pytest

#### Instalación y Lanzamiento
1. Clonar el repositorio
2. Crear un entorno virtual
3. Instalar dependencias: `pip install -r requirements.txt`
4. En terminal, inicie el frontend: `streamlit run app.py`

### Licencia

Este proyecto está disponible bajo una licencia dual:

1. Licencia Pública General Affero de GNU v3.0 (AGPL-3.0)
2. Licencia Comercial

Para más detalles, consulte el archivo [LICENSE](LICENSE) y [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

### Contribuir

Por favor, lea nuestro [CONTRIBUTING.md](CONTRIBUTING.md) para obtener detalles sobre nuestro código de conducta y el proceso para enviar pull requests.
