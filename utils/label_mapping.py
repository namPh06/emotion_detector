"""
Label mapping utilities shared by training and inference.

DAiSEE contains four affect dimensions:
- Boredom
- Engagement
- Confusion
- Frustration

This project maps them into four application labels:
0 -> Buon chan / Boring
1 -> Tap trung / Focus
2 -> Hung thu / Interested
3 -> Binh thuong / Normal
"""

EMOTION_LABELS = {
    0: "Buon chan",
    1: "Tap trung",
    2: "Hung thu",
    3: "Binh thuong",
}

EMOTION_LABELS_EN = {
    0: "Boring",
    1: "Focus",
    2: "Interested",
    3: "Normal",
}

EMOTION_COLORS = {
    0: (0, 80, 255),
    1: (0, 200, 0),
    2: (255, 165, 0),
    3: (200, 200, 200),
}

FER2013_TO_EMOTION = {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "sad": 0,
    "happy": 2,
    "surprise": 2,
    "neutral": 3,
}

# AffectNet: 0=Neutral 1=Happy 2=Sad 3=Surprise 4=Fear 5=Disgust 6=Anger 7=Contempt
AFFECTNET_TO_EMOTION = {
    1: 2,  # Happy     -> Hung thu
    2: 0,  # Sad       -> Buon chan
    3: 2,  # Surprise  -> Hung thu
    4: 0,  # Fear      -> Buon chan
    5: 0,  # Disgust   -> Buon chan
    6: 0,  # Anger     -> Buon chan
    7: 0,  # Contempt  -> Buon chan
    # Neutral (0) -> split 60% Tap trung / 40% Binh thuong (handled in training)
}


def map_daisee_to_label(boredom: int, engagement: int, confusion: int, frustration: int) -> int:
    """
    Convert DAiSEE's 4-dimensional labels to one project label.

    Priority:
    1. Boring: boredom >= 2 and engagement <= 1
    2. Interested: engagement >= 2 and confusion >= 1 and frustration < 2
    3. Focus: engagement >= 2
    4. Normal: everything else
    """
    if boredom >= 2 and engagement <= 1:
        return 0

    if engagement >= 2 and confusion >= 1 and frustration < 2:
        return 2

    if engagement >= 2:
        return 1

    return 3


def get_class_weights_info():
    return {
        "note": "DAiSEE is imbalanced. The focus class usually dominates the dataset.",
        "suggestion": "Use compute_class_weight plus a small manual boost for rare classes.",
    }


def get_ordered_labels(language: str = "vi"):
    mapping = EMOTION_LABELS if language.lower() == "vi" else EMOTION_LABELS_EN
    return [mapping[idx] for idx in sorted(mapping)]
