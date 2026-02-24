"""
Flask server for Emotion Detection web application.
"""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/")
def render_index_page():
    """
    Renders the main web page.
    """
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detection_route():
    """
    Receives text from the web UI, runs emotion detection, and returns formatted output.
    """
    text_to_analyze = request.args.get("textToAnalyze", default="", type=str)
    result = emotion_detector(text_to_analyze)

    # Task 7B: Handle blank/invalid input
    if result.get("dominant_emotion") is None:
        return "Invalid text! Please try again!."

    anger = result.get("anger")
    disgust = result.get("disgust")
    fear = result.get("fear")
    joy = result.get("joy")
    sadness = result.get("sadness")
    dominant = result.get("dominant_emotion")

    return (
        "For the given statement, the system response is "
        f"'anger': {anger}, 'disgust': {disgust}, 'fear': {fear}, "
        f"'joy': {joy} and 'sadness': {sadness}. "
        f"The dominant emotion is {dominant}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    