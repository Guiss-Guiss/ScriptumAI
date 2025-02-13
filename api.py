import logging
from flask import Flask, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
from main import RAGApplication
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from datetime import datetime, timedelta
import threading
import uuid
from collections import defaultdict
import time
from config import SUPPORTED_FILE_TYPES
from flask import Flask, request, jsonify
from main import rag_app

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Starting api.py")
logger.debug("Current working directory: %s", os.getcwd())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logger.info("Flask app initialized")

class TaskTracker:
    def __init__(self):
        self.tasks = {}
        self.embeddings_count = defaultdict(int)
        self.last_success = None
        self.lock = threading.Lock()

    def create_task(self, task_id, filename):
        with self.lock:
            self.tasks[task_id] = {
                'status': 'In Progress',
                'filename': filename,
                'start_time': datetime.now(),
                'embeddings_count': 0,
                'error': None
            }

    def update_task(self, task_id, status=None, embeddings_count=None, error=None):
        with self.lock:
            if task_id in self.tasks:
                if status:
                    self.tasks[task_id]['status'] = status
                if embeddings_count is not None:
                    self.tasks[task_id]['embeddings_count'] = embeddings_count
                if error:
                    self.tasks[task_id]['error'] = error
                if status == 'Completed':
                    self.last_success = datetime.now()

    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id, {'status': 'Not Found'})

    def cleanup_old_tasks(self, max_age_hours=24):
        with self.lock:
            current_time = datetime.now()
            to_remove = []
            for task_id, task in self.tasks.items():
                if current_time - task['start_time'] > timedelta(hours=max_age_hours):
                    to_remove.append(task_id)
            for task_id in to_remove:
                del self.tasks[task_id]

task_tracker = TaskTracker()

def cleanup_old_tasks():
    while True:
        task_tracker.cleanup_old_tasks()
        time.sleep(3600)  # Clean up every hour

@app.before_request
def start_cleanup_thread():
    if not hasattr(app, 'cleanup_thread_started'):
        cleanup_thread = threading.Thread(target=cleanup_old_tasks, daemon=True)
        cleanup_thread.start()
        app.cleanup_thread_started = True

def allowed_file(filename, content_type):
    """Check if the file extension and MIME type are allowed."""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS and content_type in SUPPORTED_FILE_TYPES

def ingest_document_thread(file_path, task_id):
    try:
        rag_app.ingest_document(file_path)
        task_tracker.update_task(task_id, status="Completed")
        logger.debug(f"File ingested successfully: {file_path}")
    except Exception as e:
        task_tracker.update_task(task_id, status=f"Failed: {str(e)}")
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
    
    if file and allowed_file(file.filename, file.content_type):
        try:
            task_id = str(uuid.uuid4())
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                logger.debug(f"Temporary file created: {temp_file.name}")
                
                task_tracker.create_task(task_id, file.filename)
                thread = threading.Thread(target=ingest_document_thread, args=(temp_file.name, task_id))
                thread.start()
                
            logger.debug(f"Ingestion task started for file: {file.filename}")
            return jsonify({
                "message": "File ingestion task started",
                "task_id": task_id
            }), 202
        except Exception as e:
            logger.error(f"Error starting ingestion task: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    else:
        logger.warning("File type not allowed")
        return jsonify({"error": "File type not allowed"}), 400


@app.route('/api/ingestion_status/<task_id>', methods=['GET'])
def check_ingestion_status(task_id):
    task = task_tracker.get_task(task_id)
    return jsonify({
        "status": task.get('status'),
        "embeddings_count": task.get('embeddings_count', 0),
        "filename": task.get('filename'),
        "error": task.get('error')
    })

@app.route('/api/query', methods=['POST'])
def process_query():
    logger.debug(f"Received query request: {request.get_json()}")
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            logger.warning("No query provided in request")
            return jsonify({"error": "No query provided"}), 400

        query = data['query']
        
        response = rag_app.query_component.process_query(query)
        
        logger.debug(f"Query processed successfully: {response}")
        return jsonify({
            "response": response,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = rag_app.get_stats()
        stats.update({
            "recent_success": task_tracker.last_success is not None and 
                            (datetime.now() - task_tracker.last_success) < timedelta(minutes=5),
            "active_tasks": sum(1 for task in task_tracker.tasks.values() 
                              if task['status'] == 'In Progress')
        })
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint called")
    try:
        chroma_health = rag_app.ingest_component.check_chroma_health()
        recent_success = (task_tracker.last_success is not None and 
                        (datetime.now() - task_tracker.last_success) < timedelta(minutes=5))
        
        return jsonify({
            'status': 'healthy' if chroma_health else 'unhealthy',
            'database': 'connected' if chroma_health else 'disconnected',
            'recent_success': recent_success,
            'active_tasks': len([t for t in task_tracker.tasks.values() 
                               if t['status'] == 'In Progress'])
        }), 200 if chroma_health else 503
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/api/query', methods=['POST'])
def query_endpoint():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data['query']
        response = rag_app.query_component.process_query(query)

        if response.get("error"):
            return jsonify({"error": response['error']}), 500

        return jsonify({"response": response, "status": "success"})
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=False)
