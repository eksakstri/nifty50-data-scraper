from flask import Flask, jsonify
import subprocess
from datetime import datetime
import sys

print(sys.executable)

app = Flask(__name__)

SCRIPTS = [
    "nifty_50_scraper.py",
    "market_summary_generator.py",
    "option_chain_scraper.py",
    "pdf_downloader.py"
]


def run_pipeline():

    results = []

    print(f"\n[{datetime.now()}] Starting pipeline")

    for script in SCRIPTS:

        print(f"Running {script}")

        try:

            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True
            )

            results.append(
                {
                    "script": script,
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "status": (
                        "success"
                        if result.returncode == 0
                        else "failed"
                    )
                }
            )

        except Exception as e:

            results.append(
                {
                    "script": script,
                    "status": "error",
                    "error": str(e)
                }
            )

    print(f"\n[{datetime.now()}] Pipeline finished")
    return results


@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "healthy"
        }
    )


@app.route("/run-ingestion", methods=["POST"])
def trigger_pipeline():
    results = run_pipeline()
    return jsonify(
        {
            "success": True,
            "timestamp": str(datetime.now()),
            "results": results
        }
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )
