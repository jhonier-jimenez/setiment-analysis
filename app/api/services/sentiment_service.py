import tensorflow as tf
import numpy as np
from typing import Tuple

class SentimentService:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def analyze_sentiment(self, text: str) -> Tuple[str, float, bool]:
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=self.model.input_shape[1],
            padding='post',
            truncating='post'
        )
        
        prediction = self.model.predict(padded_sequence)[0][0]
        confidence = float(prediction)
        is_positive = confidence >= 0.5
        sentiment = "positive" if is_positive else "negative"
        
        return sentiment, confidence, is_positive 