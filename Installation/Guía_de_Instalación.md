# Guía de Instalación Completa de ScriptumAI

Esta guía le acompañará a través de todo el proceso de instalación de ScriptumAI, incluyendo todos los prerrequisitos y componentes necesarios.

## Tabla de Contenidos
1. [Instalar Python 12](#1-instalar-python-12)
2. [Instalar CUDA Toolkit (para uso de GPU)](#2-instalar-cuda-toolkit-para-uso-de-gpu)
3. [Instalar Ollama](#3-instalar-ollama)
4. [Descargar los Modelos Requeridos](#4-descargar-los-modelos-requeridos)
5. [Configurar ScriptumAI](#5-configurar-scriptum-ai)
6. [Ejecutar ScriptumAI](#6-ejecutar-scriptum-ai)

## 1. Instalar Python 12

Primero, verifique si Python 12 ya está instalado:

```bash
python --version
```

Si Python 12 no está instalado:

1. Visite https://www.python.org/downloads/
2. Descargue Python 12.x.x
3. Ejecute el instalador, asegurándose de marcar "Agregar Python al PATH"
4. Verifique la instalación ejecutando `python --version` en una nueva terminal

## 2. Instalar CUDA Toolkit (para uso de GPU)

Si tiene una GPU NVIDIA compatible y desea usarla para procesamiento acelerado:

1. Visite https://developer.nvidia.com/cuda-downloads
2. Seleccione su sistema operativo y siga las instrucciones de instalación
3. Verifique la instalación ejecutando `nvcc --version` en una terminal

## 3. Instalar Ollama

### Para macOS y Linux:

```bash
curl https://ollama.ai/install.sh | sh
```

### Para Windows:

1. Visite https://ollama.com/download/windows
2. Descargue y ejecute el último instalador de Windows

Verifique la instalación:

```bash
ollama --version
```

## 4. Descargar los Modelos Requeridos

Descargue los modelos necesarios para ScriptumAI:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verifique que los modelos estén instalados:

```bash
ollama list
```

## 5. Configurar ScriptumAI

1. Clone el repositorio de ScriptumAI (reemplace con la URL real del repositorio):

```bash
git clone https://github.com/Guiss-Guiss/ScriptumAI.git
cd ScriptumAI
```

2. Cree un entorno virtual:

```bash
python -m venv scriptum
```

3. Active el entorno virtual:

- En Windows:
  ```
  scriptum\Scripts\activate
  ```
- En macOS y Linux:
  ```
  source scriptum/bin/activate
  ```

4. Instale las dependencias:

```bash
pip install -r requirements.txt
```
- En MacOs :
```bash
brew install libmagic
```

## 6. Ejecutar ScriptumAI

1. En una terminal, asegúrese de estar en el directorio ScriptumAI y que su entorno virtual esté activado, luego ejecute:

```bash
python api.py
```

2. Abra otra terminal, navegue al directorio ScriptumAI, active el entorno virtual y ejecute:

```bash
streamlit run app.py
```

Esto iniciará la aplicación ScriptumAI. Puede acceder a la interfaz de usuario abriendo un navegador web y navegando a la URL proporcionada por Streamlit (generalmente http://localhost:8501).

## Solución de Problemas

- Si encuentra errores de "comando no encontrado", asegúrese de que la herramienta relevante esté correctamente instalada y agregada al PATH de su sistema.
- Para problemas relacionados con la GPU, asegúrese de que sus controladores NVIDIA estén actualizados y sean compatibles con el CUDA Toolkit instalado.
- Si encuentra problemas con Ollama o la descarga de modelos, verifique su conexión a internet y la configuración de su firewall.
- Para cualquier problema de instalación de paquetes Python, asegúrese de estar usando la versión correcta de pip dentro de su entorno virtual.

Si continúa experimentando problemas, consulte la documentación oficial de cada componente o busque ayuda en los foros de la comunidad ScriptumAI.
