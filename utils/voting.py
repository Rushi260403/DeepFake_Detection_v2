import numpy as np

FAKE_THRESHOLD = 0.60
REAL_THRESHOLD = 0.35

def final_decision(fake_probs):
    if len(fake_probs) < 5:
        return "FACE NOT FOUND", 0.0

    fake_probs = np.array(fake_probs)

    mean_fake = fake_probs.mean()
    strong_fake = np.sum(fake_probs >= FAKE_THRESHOLD)
    ratio_fake = strong_fake / len(fake_probs)

    # 🔥 STRONG FAKE SIGNAL
    if mean_fake >= 0.45 or ratio_fake >= 0.30:
        confidence = min(95.0, mean_fake * 100)
        return "FAKE", confidence

    # 🔥 STRONG REAL SIGNAL
    if mean_fake <= REAL_THRESHOLD:
        confidence = min(95.0, (1 - mean_fake) * 100)
        return "REAL", confidence

    # 🔥 AMBIGUOUS / AI-GEN
    return "UNCERTAIN", 50.0
