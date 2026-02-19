import numpy as np

FAKE_THRESHOLD = 0.60

def final_decision(fake_probs):
    fake_probs = np.array(fake_probs)

    if len(fake_probs) < 8:
        return "FACE NOT FOUND", 0.0

    mean_fake = fake_probs.mean()

    high_fake_ratio = np.sum(fake_probs >= 0.60) / len(fake_probs)
    low_fake_ratio  = np.sum(fake_probs <= 0.25) / len(fake_probs)

    # 🔥 1. STRONG FAKE (unchanged, strict)
    if high_fake_ratio >= 0.35 or mean_fake >= 0.55:
        confidence = min(95.0, mean_fake * 100)
        return "FAKE", confidence

    # ✅ 2. STRONG REAL (CONSISTENCY-BASED, NOT MEAN-BASED)
    if low_fake_ratio >= 0.70:
        confidence = min(95.0, (1 - mean_fake) * 100)
        return "REAL", confidence

    # 🤖 3. EVERYTHING ELSE
    return "UNCERTAIN", 50.0
