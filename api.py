import logging
from flask import Flask, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
from main import RAGApplication
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from datetime import datetime
import threading
import uuid
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Starting api.py")
logger.debug("Current working directory: %s", os.getcwd())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
rag_app = RAGApplication()

logger.info("Flask app initialized")

ingestion_tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ingest_document_worker(file_path, task_id):
    try:
        rag_app.ingest_document(file_path)
        ingestion_tasks[task_id] = "Completed"
        logger.debug(f"File ingested successfully: {file_path}")
    except Exception as e:
        ingestion_tasks[task_id] = f"Failed: {str(e)}"
        logger.error(f"Error ingesting document: {str(e)}", exc_info=True)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/api/ingest', methods=['POST'])
def ingest_document():
    logger.debug(f"Received ingest request. Files: {request.files}")
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    logger.debug(f"File details - filename: {file.filename}, content_type: {file.content_type}")
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                logger.debug(f"Temporary file created: {temp_file.name}")
                task_id = str(uuid.uuid4())
                ingestion_tasks[task_id] = "In Progress"
                process = mp.Process(target=ingest_document_worker, args=(temp_file.name, task_id))
                process.start()
            logger.debug(f"Ingestion task started for file: {file.filename}")
            return jsonify({"message": "File ingestion task started", "task_id": task_id}), 202
        except Exception as e:
            logger.error(f"Error starting ingestion task: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    else:
        logger.warning("File type not allowed")
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/ingestion_status/<task_id>', methods=['GET'])
def check_ingestion_status(task_id):
    status = ingestion_tasks.get(task_id, "Not Found")
    return jsonify({"status": status})

@app.route('/api/ingested_files', methods=['GET'])
def get_ingested_files():
    try:
        files = rag_app.get_ingested_files()
        return jsonify(files)
    except Exception as e:
        logger.error(f"Error getting ingested files: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/remove', methods=['POST'])
def remove_document():
    data = request.json
    if not data or 'file_name' not in data:
        return jsonify({"error": "No file name provided"}), 400
    try:
        file_name = data['file_name']
        success = rag_app.remove_document(file_name)
        if success:
            return jsonify({"message": f"File {file_name} removed successfully"}), 200
        else:
            return jsonify({"error": f"File {file_name} not found or could not be removed"}), 404
    except Exception as e:
        logger.error(f"Error removing document: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    try:
        result = rag_app.process_query(data['query'])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def semantic_search():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    try:
        k = data.get('k', 5)
        results = rag_app.semantic_search(data['query'], k)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error performing semantic search: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = rag_app.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint called")
    try:
        chroma_health = rag_app.ingest_component.check_chroma_health()
        logger.info(f"Chroma health check result: {chroma_health}")
        return jsonify({
            'status': 'healthy' if chroma_health else 'unhealthy',
            'database': 'connected' if chroma_health else 'disconnected'
        }), 200 if chroma_health else 503
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': str(e)
        }), 503

@app.route('/api/logs', methods=['GET'])
def get_logs():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(BASE_DIR, "rag_app.log")

        if not os.path.exists(log_file_path):
            open(log_file_path, 'a').close()
            logger.info("Log file created at: %s", log_file_path)

        with open(log_file_path, 'r') as log_file:
            logs = log_file.readlines()

        formatted_logs = []
        for log in logs:
            try:
                timestamp, level, message = log.split(" | ", 2)
                formatted_logs.append({
                    "timestamp": timestamp,
                    "level": level.strip(),
                    "message": message.strip()
                })
            except ValueError:
                formatted_logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": log.strip()
                })

        return jsonify(formatted_logs)
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=5000, debug=True)