
# ScriptumAI | Guía de instalación completa

Esta guía te guiará a través de todo el proceso de configuración de ScriptumAI, incluidos todos los prerrequisitos y componentes necesarios.

## Tabla de contenidos
- [ScriptumAI | Guía de instalación completa](#scriptumai--guía-de-instalación-completa)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [1. Instalar Python 11](#1-instalar-python-11)
  - [2. Instalar Ollama](#2-instalar-ollama)
    - [Para macOS y Linux:](#para-macos-y-linux)
    - [Para Windows:](#para-windows)
  - [3. Descargar los modelos requeridos](#3-descargar-los-modelos-requeridos)
  - [4. Configurar ScriptumAI](#4-configurar-scriptumai)
  - [5. Ejecutar ScriptumAI](#5-ejecutar-scriptumai)
  - [Resolución de problemas](#resolución-de-problemas)

## 1. Instalar Python 11

Primero, verifica si Python 11 ya está instalado:

```bash
python --version
```

Si Python 11 no está instalado:

1. Ve a https://www.python.org/downloads/
2. Descarga Python 11.x.x
3. Ejecuta el instalador asegurándote de marcar "Agregar Python al PATH"
4. Verifica la instalación ejecutando `python --version` en una nueva terminal

## 2. Instalar Ollama

### Para macOS y Linux:

```bash
curl https://ollama.ai/install.sh | sh
```

### Para Windows:

1. Ve a https://ollama.com/download/OllamaSetup.exe
2. Descarga y ejecuta el instalador más reciente para Windows

Verifica la instalación:

```bash
ollama --version
```

## 3. Descargar los modelos requeridos

Descarga los modelos necesarios para ScriptumAI desde https://ollama.com/search

```bash
ollama pull nomic-embed-text  # Requerido para embeddings
ollama pull llama3.2          # LLM predeterminado (o cualquier modelo que prefieras)
```

**Nota:** Puedes instalar múltiples modelos LLM. ScriptumAI incluye un **selector de modelo** en la sección de Consulta que te permite elegir entre todos tus modelos Ollama instalados en tiempo de ejecución. El modelo `nomic-embed-text` es requerido para los embeddings, pero puedes usar cualquier LLM para las respuestas a consultas.

Verifica que los modelos estén instalados:

```bash
ollama list
```

## 4. Configurar ScriptumAI

1. Clona el repositorio de ScriptumAI:

```bash
git clone https://github.com/Guiss-Guiss/ScriptumAI-CPU-.git
cd ScriptumAI
```

2. Crea un entorno virtual:

```bash
python -m venv scriptum python=3.11
```

3. Activa el entorno virtual:

- En Windows:
  ```
  scriptum\Scriptsctivate
  ```
- En macOS y Linux:
  ```
  source scriptum/bin/activate
  ```

4. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## 5. Ejecutar ScriptumAI

2. Abre una terminal, navega al directorio ScriptumAICPU, activa el entorno virtual y ejecuta:

```bash
streamlit run app.py
```

Esto iniciará la aplicación ScriptumAI. Puedes acceder a la interfaz de usuario abriendo un navegador web y navegando a la URL proporcionada por Streamlit (normalmente http://localhost:8501).

## Resolución de problemas

- Si encuentras errores de "comando no encontrado", asegúrate de que la herramienta relevante esté instalada correctamente y agregada al PATH de tu sistema.
- Si tienes problemas con Ollama o las descargas de modelos, verifica tu conexión a Internet y la configuración del firewall.
- Para cualquier problema de instalación de paquetes de Python, asegúrate de que estás usando la versión correcta de pip dentro de tu entorno virtual.

Si los problemas persisten, consulta la documentación oficial de cada componente.
