
# ScriptumAI | Guide d'installation complet

Ce guide vous guidera tout au long du processus d'installation de ScriptumAI, y compris tous les prérequis et composants nécessaires.

## Table des matières
- [ScriptumAI | Guide d'installation complet](#scriptumai--guide-dinstallation-complet)
  - [Table des matières](#table-des-matières)
  - [1. Installer Python 11](#1-installer-python-11)
  - [2. Installer Ollama](#2-installer-ollama)
    - [Pour macOS et Linux :](#pour-macos-et-linux-)
    - [Pour Windows :](#pour-windows-)
  - [3. Télécharger les modèles requis](#3-télécharger-les-modèles-requis)
  - [4. Configurer ScriptumAI](#4-configurer-scriptumai)
  - [5. Exécuter ScriptumAI](#5-exécuter-scriptumai)
  - [Dépannage](#dépannage)

## 1. Installer Python 11

Tout d'abord, vérifiez si Python 11 est déjà installé :

```bash
python --version
```

Si Python 11 n'est pas installé :

1. Rendez-vous sur https://www.python.org/downloads/
2. Téléchargez Python 11.x.x
3. Exécutez l'installateur en veillant à cocher "Ajouter Python au PATH"
4. Vérifiez l'installation en exécutant `python --version` dans un nouveau terminal

## 2. Installer Ollama

### Pour macOS et Linux :

```bash
curl https://ollama.ai/install.sh | sh
```

### Pour Windows :

1. Rendez-vous sur https://ollama.com/download/windows
2. Téléchargez et exécutez le dernier installateur pour Windows

Vérifiez l'installation :

```bash
ollama --version
```

## 3. Télécharger les modèles requis

Téléchargez les modèles nécessaires pour ScriptumAI : https://ollama.com/search pour la liste.

```bash
ollama pull llama3.1:latest  # Ceci est un exemple. Vous pouvez télécharger plus d'un modèle.
ollama pull nomic-embed-text # Nécessaire.
```

Vérifiez que les modèles sont installés :

```bash
ollama list
```

## 4. Configurer ScriptumAI

1. Clonez le dépôt ScriptumAI :

```bash
git clone https://github.com/Guiss-Guiss/ScriptumAI-CPU-.git
cd ScriptumAI
```

2. Créez un environnement virtuel :

```bash
python -m venv scriptum python=3.11
```

3. Activez l'environnement virtuel :

- Sur Windows :
  ```
  scriptum\Scripts\Activate
  ```
- Sur macOS et Linux :
  ```
  source scriptum/bin/activate
  ```

4. Installez les dépendances :

```bash
pip install -r requirements.txt
```

## 5. Exécuter ScriptumAI

2. Ouvrez un terminal, naviguez vers le répertoire ScriptumAICPU, activez l'environnement virtuel et exécutez :

```bash
streamlit run app.py
```

Cela démarrera l'application ScriptumAI. Vous pouvez accéder à l'interface utilisateur en ouvrant un navigateur web et en naviguant vers l'URL fournie par Streamlit (généralement http://localhost:8501).

## Dépannage

- Si vous rencontrez des erreurs "commande introuvable", assurez-vous que l'outil pertinent est correctement installé et ajouté au PATH de votre système.
- Si vous avez des problèmes avec Ollama ou le téléchargement des modèles, vérifiez votre connexion Internet et les paramètres du pare-feu.
- Pour tout problème d'installation de package Python, assurez-vous que vous utilisez la bonne version de pip dans votre environnement virtuel.

Si les problèmes persistent, veuillez consulter la documentation officielle de chaque composant.
