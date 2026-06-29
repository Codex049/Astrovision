from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from Architecture import process
import os
import csv
import numpy as np
import requests
import json

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/upload": {"origins": "*"},
    r"/chat": {"origins": "*"},
    r"/chat/status": {"origins": "*"},
    r"/example images/*": {"origins": "*"}
})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OLLAMA CONFIGURATION — change these to swap models
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT  = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

SYSTEM_PROMPT = """You are AstroVision AI, an expert astrophysicist and data scientist specializing in the Sloan Digital Sky Survey (SDSS). You help users understand astronomical observations in plain, accessible language.

Your expertise includes:
- Interpreting UGRIZ photometric magnitudes (u=ultraviolet, g=green, r=red, i=near-infrared, z=far-infrared)
- Galaxy morphology classification (elliptical, spiral, irregular, etc.)
- Star vs galaxy classification and the physical properties that distinguish them
- Color-magnitude diagrams and what an object's position reveals about its age, temperature, composition, and distance
- Redshift estimation from photometric colors
- Stellar classification (spectral types O, B, A, F, G, K, M)

When the user uploads an image, you will receive the CNN classification results and UGRIZ magnitudes as context. Use this data to provide insightful, educational analysis. Explain what the numbers mean physically — for example, what the color index (g-r) tells us about temperature, or what the magnitude values suggest about brightness and distance.

Keep responses concise (2-4 paragraphs max), conversational, and educational. Avoid unnecessary jargon, but when you use technical terms, briefly explain them. If the user asks something outside the scope of the data provided, be honest about the limitations."""


def check_ollama_available():
    """Check if Ollama is running and the model is available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, "Ollama is running but returned an unexpected status."

        models = resp.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        # Check exact match or match without tag
        base_name = OLLAMA_MODEL.split(":")[0]
        for name in model_names:
            if name == OLLAMA_MODEL or name.startswith(base_name):
                return True, None

        return False, (
            f"Model '{OLLAMA_MODEL}' is not installed. "
            f"Available models: {', '.join(model_names) if model_names else 'none'}. "
            f"Run: ollama pull {OLLAMA_MODEL}"
        )
    except requests.ConnectionError:
        return False, (
            "Cannot connect to Ollama. Make sure it is installed and running. "
            "Start it with: ollama serve"
        )
    except Exception as e:
        return False, f"Ollama health check failed: {str(e)}"


# Run health check at startup (non-blocking — just prints a warning)
_available, _warning = check_ollama_available()
if _available:
    print(f"✅ Ollama connected — using model '{OLLAMA_MODEL}'")
else:
    print(f"⚠️  Ollama not ready: {_warning}")
    print("   Chat will gracefully degrade until Ollama is available.")


# CMD reference data
CMD_BACKGROUND = []
try:
    with open("references.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 500:
                break
            try:
                g = float(row.get("g", row.get("G", 0)))
                r = float(row.get("r", row.get("R", 0)))
                u = float(row.get("u", row.get("U", g + 1.2)))
                i_val = float(row.get("i", row.get("I", r - 0.4)))
                z = float(row.get("z", row.get("Z", r - 0.8)))
                CMD_BACKGROUND.append({
                    "u": u, "g": g, "r": r, "i": i_val, "z": z,
                    "color_index": g - r
                })
            except Exception:
                continue
    print(f"Loaded {len(CMD_BACKGROUND)} CMD reference galaxies.")
except FileNotFoundError:
    print("WARNING: references.csv not found!")
except Exception as e:
    print(e)


def sanitize_for_json(data):
    """Recursively converts NumPy data types to native Python types for JSON."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data


# Routes

@app.route("/")
def index():
    return send_from_directory(".", "site.html")


@app.route("/example images/<path:filename>")
def serve_example_images(filename):
    return send_from_directory("example images", filename)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        results = process(filepath)
        safe_results = sanitize_for_json(results)
        safe_results["cmd_background"] = CMD_BACKGROUND
        return jsonify(safe_results)
    except Exception as e:
        print(f"Inference Error: {e}")
        return jsonify({"error": str(e)}), 500


# Chat endpoint — Ollama with graceful fallback

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True)
    if not body or "message" not in body:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = body["message"]
    astro_context = body.get("astro_context", {})
    conversation_history = body.get("history", [])

    # Build a context block from the CNN results so the LLM knows what was analyzed
    context_lines = []
    if astro_context:
        obj_type = astro_context.get("object_type", "unknown")
        probs = astro_context.get("star_galaxy_probs", {})
        ugriz = astro_context.get("ugriz", [])
        morphology = astro_context.get("morphology_probs", None)

        context_lines.append(f"=== Current Analysis Results ===")
        context_lines.append(f"CNN Classification: {obj_type}")
        context_lines.append(f"Galaxy probability: {probs.get('galaxy', '?')}%  |  Star probability: {probs.get('star', '?')}%")

        if ugriz and len(ugriz) == 5:
            u, g, r, i, z = ugriz
            context_lines.append(f"UGRIZ magnitudes: u={u:.3f}, g={g:.3f}, r={r:.3f}, i={i:.3f}, z={z:.3f}")
            context_lines.append(f"Color index (g-r): {(g - r):.3f}")

        if morphology:
            morph_str = ", ".join(f"{k}: {v}%" for k, v in morphology.items())
            context_lines.append(f"Morphology: {morph_str}")

    context_block = "\n".join(context_lines)

    # Build the messages array for Ollama
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # If we have context, inject it as an early system message
    if context_block:
        messages.append({
            "role": "system",
            "content": f"The user has uploaded an astronomical image. Here are the analysis results:\n\n{context_block}\n\nUse this data to answer the user's questions."
        })

    # Append conversation history (already in [{role, content}] format)
    for msg in conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Current user message
    messages.append({"role": "user", "content": user_message})

    # Call Ollama
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )

        if resp.status_code != 200:
            error_detail = resp.text[:300]
            return jsonify({
                "reply": f"Ollama returned an error (HTTP {resp.status_code}). Detail: {error_detail}",
                "error": True
            }), 200  # Return 200 so frontend can display the message gracefully

        data = resp.json()
        reply = data.get("message", {}).get("content", "")

        if not reply:
            return jsonify({
                "reply": "The model returned an empty response. Try rephrasing your question.",
                "error": True
            }), 200

        return jsonify({"reply": reply, "error": False})

    except requests.ConnectionError:
        return jsonify({
            "reply": "Cannot reach Ollama — it may not be running on this machine. Start it with 'ollama serve' in a terminal.",
            "error": True
        }), 200

    except requests.Timeout:
        return jsonify({
            "reply": f"Ollama timed out after {OLLAMA_TIMEOUT}s. The model may be loading for the first time or the machine is under heavy load.",
            "error": True
        }), 200

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({
            "reply": f"Something went wrong with the chat service: {str(e)}",
            "error": True
        }), 200


# Ollama health check endpoint (for frontend status)

@app.route("/health")
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": OLLAMA_MODEL,
        "ollama_available": _available
    })


@app.route("/debug")
def debug():
    """Debug endpoint to check request headers and connectivity."""
    return jsonify({
        "host": request.host,
        "remote_addr": request.remote_addr,
        "user_agent": request.user_agent.string,
        "scheme": request.scheme,
        "base_url": request.base_url,
        "server_info": {
            "ollama_base_url": OLLAMA_BASE_URL,
            "ollama_model": OLLAMA_MODEL,
            "ollama_available": _available,
            "references_loaded": len(CMD_BACKGROUND)
        }
    })


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        ip_addr = socket.gethostbyname(hostname)
    except:
        ip_addr = "your-machine-ip"
    
    print(f"\n{'='*60}")
    print(f"AstroVision AI is running")
    print(f"{'='*60}")
    print(f"Local access:     http://127.0.0.1:5001")
    print(f"Network access:   http://{ip_addr}:5001")
    print(f"Debug info:       http://{ip_addr}:5001/debug")
    print(f"Health check:     http://{ip_addr}:5001/health")
    print(f"\n💡 CORS enabled for cross-origin requests")
    print(f"{'='*60}\n")
    
    app.run(debug=True, host="0.0.0.0", port=5001)