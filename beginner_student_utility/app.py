import os

from dotenv import load_dotenv
load_dotenv()
from flask import Flask, flash, render_template, request

from llm_client import PROMPTS, run_task

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

TASK_LABELS = {
    "summarize": "Summarize Notes",
    "quiz": "Generate Quiz",
    "improve_answer": "Improve an Answer",
    "explain": "Explain a Concept",
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    selected_task = "summarize"
    input_text = ""

    if request.method == "POST":
        selected_task = request.form.get("task", "summarize")
        input_text = request.form.get("content", "")

        if not input_text.strip():
            flash("Please enter some text before submitting.", "error")
        elif selected_task not in PROMPTS:
            flash("Please choose a valid task type.", "error")
        else:
            try:
                result = run_task(selected_task, input_text)
                if not result.strip():
                    flash("The model returned an empty response — please try again.", "error")
                    result = None
            except Exception as e:
                flash(f"Something went wrong generating a response: {e}", "error")

    return render_template(
        "index.html",
        tasks=TASK_LABELS,
        selected_task=selected_task,
        input_text=input_text,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
