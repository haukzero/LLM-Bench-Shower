import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from runner import get_llm_bench_runner

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the runner
runner = get_llm_bench_runner()

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get available datasets."""
    return jsonify(runner.available_datasets)

@app.route('/api/submit', methods=['POST'])
def submit_task():
    """Submit benchmark tasks."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    if not isinstance(data, list):
        data = [data]

    requests_to_submit = []
    for item in data:
        # Generate req_id if not present
        if 'req_id' not in item:
            item['req_id'] = str(uuid.uuid4())
        
        # Convert model_type to bytes as required by runner
        if 'model_type' in item and isinstance(item['model_type'], str):
            item['model_type'] = item['model_type'].encode('utf-8')
        
        requests_to_submit.append(item)

    try:
        runner.submit_requests(requests_to_submit)
        return jsonify({
            "status": "success",
            "submitted_count": len(requests_to_submit),
            "req_ids": [r['req_id'] for r in requests_to_submit]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get benchmark results (polling)."""
    count = request.args.get('count', type=int)
    timeout = request.args.get('timeout', default=0.1, type=float)
    
    results = runner.get_results(count=count, timeout=timeout)
    
    # Convert NamedTuple to dict for JSON serialization
    json_results = []
    for res in results:
        json_results.append({
            "req_id": res.req_id,
            "result": res.result,
            "error": res.error
        })
        
    return jsonify(json_results)

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all benchmark history."""
    history = runner.get_all_history()
    # history is list of tuples: (model_name, dataset_name, results, created_at, updated_at)
    formatted_history = []
    for item in history:
        formatted_history.append({
            "model_name": item[0],
            "dataset_name": item[1],
            "results": item[2],
            "created_at": item[3],
            "updated_at": item[4]
        })
    return jsonify(formatted_history)

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear benchmark history."""
    model_name = request.args.get('model_name')
    dataset_name = request.args.get('dataset_name')
    
    try:
        count = runner.clear_history(model_name=model_name, dataset_name=dataset_name)
        return jsonify({"status": "success", "deleted_count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics."""
    return jsonify(runner.get_database_stats())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
