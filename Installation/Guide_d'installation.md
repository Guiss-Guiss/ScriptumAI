# Guide d'Installation Complet de ScriptumAI

Ce guide vous accompagnera à travers l'ensemble du processus d'installation de ScriptumAI, y compris tous les prérequis et les composants nécessaires.

## Table des Matières
- [Guide d'Installation Complet de ScriptumAI](#guide-dinstallation-complet-de-scriptumai)
  - [Table des Matières](#table-des-matières)
  - [1. Installer Python 12](#1-installer-python-12)
  - [2. Installer CUDA Toolkit (pour l'utilisation du GPU)](#2-installer-cuda-toolkit-pour-lutilisation-du-gpu)
  - [3. Installer Ollama](#3-installer-ollama)
    - [Pour macOS et Linux :](#pour-macos-et-linux-)
    - [Pour Windows :](#pour-windows-)
  - [4. Télécharger les Modèles Requis](#4-télécharger-les-modèles-requis)
  - [5. Configurer ScriptumAI](#5-configurer-scriptumai)
  - [6. Lancer ScriptumAI](#6-lancer-scriptumai)
  - [Dépannage](#dépannage)

## 1. Installer Python 12

Tout d'abord, vérifiez si Python 12 est déjà installé :

```bash
python --version
```

Si Python 12 n'est pas installé :

1. Visitez https://www.python.org/downloads/
2. Téléchargez Python 12.x.x
3. Exécutez l'installateur, en vous assurant de cocher "Ajouter Python au PATH"
4. Vérifiez l'installation en exécutant `python --version` dans un nouveau terminal

## 2. Installer CUDA Toolkit (pour l'utilisation du GPU)

Si vous avez un GPU NVIDIA compatible et que vous souhaitez l'utiliser pour un traitement accéléré :

1. Visitez https://developer.nvidia.com/cuda-downloads
2. Sélectionnez votre système d'exploitation et suivez les instructions d'installation
3. Vérifiez l'installation en exécutant `nvcc --version` dans un terminal

## 3. Installer Ollama

### Pour macOS et Linux :

```bash
curl https://ollama.ai/install.sh | sh
```

### Pour Windows :

1. Visitez https://ollama.com/download/windows
2. Téléchargez et exécutez le dernier installateur Windows

Vérifiez l'installation :

```bash
ollama --version
```

## 4. Télécharger les Modèles Requis

Téléchargez les modèles nécessaires pour ScriptumAI :

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Vérifiez que les modèles sont installés :

```bash
ollama list
```

## 5. Configurer ScriptumAI

1. Clonez le dépôt ScriptumAI :

```bash
git clone https://github.com/Guiss-Guiss/ScriptumAI.git
cd ScriptumAI
```

2. Créez un environnement virtuel :

```bash
python -m venv scriptum python=3.12
```
Ou
```bash
python3 -m venv scriptum  python=3.12
```

3. Activez l'environnement virtuel :

- Sur Windows :
  ```
  scriptum\Scripts\activate
  ```
- Sur macOS et Linux :
  ```
  source scriptum/bin/activate
  ```

4. Installez les dépendances :

```bash
pip install -r requirements.txt
```
 - MacOS 
```bash
brew install libmagic
```


## 6. Lancer ScriptumAI

1. Dans un terminal, assurez-vous d'être dans le répertoire ScriptumAI et que votre environnement virtuel est activé, puis exécutez :

```bash
python api.py
```

2. Ouvrez un autre terminal, naviguez vers le répertoire ScriptumAI, activez l'environnement virtuel, et exécutez :

```bash
streamlit run app.py
```

Cela lancera l'application ScriptumAI. Vous pouvez accéder à l'interface utilisateur en ouvrant un navigateur web et en naviguant vers l'URL fournie par Streamlit (généralement http://localhost:8501).

## Dépannage

- Si vous rencontrez des erreurs "commande non trouvée", assurez-vous que l'outil concerné est correctement installé et ajouté au PATH de votre système.
- Pour les problèmes liés au GPU, assurez-vous que vos pilotes NVIDIA sont à jour et compatibles avec le CUDA Toolkit installé.
- Si vous rencontrez des problèmes avec Ollama ou le téléchargement des modèles, vérifiez votre connexion internet et les paramètres de votre pare-feu.
- Pour tout problème d'installation de paquets Python, assurez-vous d'utiliser la bonne version de pip dans votre environnement virtuel.

Si vous continuez à rencontrer des problèmes, veuillez consulter la documentation officielle de chaque composant.
