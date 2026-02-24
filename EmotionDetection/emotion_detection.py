"""
Emotion detection module using Watson NLP (Embeddable AI libraries endpoint).

This module provides a function `emotion_detector` that:
- Calls Watson EmotionPredict endpoint
- Extracts emotion scores (anger, disgust, fear, joy, sadness)
- Computes the dominant emotion (highest score)
- Handles invalid/blank input (HTTP 400) by returning None values
"""

import json
from typing import Any, Dict, Optional

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}


def _get_dominant_emotion(scores: Dict[str, float]) -> str:
    """
    Return the emotion name with the highest score.

    Args:
        scores: A dict mapping emotion names to their scores.

    Returns:
        The name of the dominant emotion.
    """
    return max(scores, key=scores.get)


def emotion_detector(text_to_analyze: str) -> Dict[str, Optional[float]]:
    """
    Analyze the given text and return emotion scores with dominant emotion.

    For blank/invalid input, the Watson service responds with HTTP 400.
    In that case, this function returns a dictionary with all values as None.

    Args:
        text_to_analyze: The input text to analyze.

    Returns:
        Dictionary containing emotion scores and dominant emotion.
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(URL, headers=HEADERS, json=payload, timeout=30)

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response_dict: Dict[str, Any] = json.loads(response.text)

    emotions = response_dict["emotionPredictions"][0]["emotion"]
    anger = float(emotions["anger"])
    disgust = float(emotions["disgust"])
    fear = float(emotions["fear"])
    joy = float(emotions["joy"])
    sadness = float(emotions["sadness"])

    scores: Dict[str, float] = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
    }

    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": _get_dominant_emotion(scores),
    }