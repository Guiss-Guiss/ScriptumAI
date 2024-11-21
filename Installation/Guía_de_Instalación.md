
# ScriptumAI | Guía de instalación completa

Esta guía te guiará a través de todo el proceso de configuración de ScriptumAI, incluidos todos los prerrequisitos y componentes necesarios.

## Tabla de contenidos
1. [Instalar Python 11](#1-instalar-python-11)
2. [Instalar Ollama](#2-instalar-ollama)
4. [Descargar los modelos requeridos](#4-descargar-los-modelos-requeridos)
5. [Configurar ScriptumAI](#5-configurar-scriptum-ai)
6. [Ejecutar ScriptumAI](#6-ejecutar-scriptum-ai)

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

1. Ve a https://ollama.com/download/windows
2. Descarga y ejecuta el instalador más reciente para Windows

Verifica la instalación:

```bash
ollama --version
```

## 3. Descargar los modelos requeridos

Descarga los modelos necesarios para ScriptumAI:

```bash
ollama pull llama3.1:latest
ollama pull nomic-embed-text
```

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
