def final_decision(fake_probs):
    """
    fake_probs : list of probabilities (0–1) for FAKE class
    """

    if len(fake_probs) == 0:
        return "UNCERTAIN", 0.0

    avg_prob = sum(fake_probs) / len(fake_probs)

    if avg_prob >= 0.7:
        return "FAKE", avg_prob
    elif avg_prob <= 0.4:
        return "REAL", avg_prob
    else:
        return "UNCERTAIN", avg_prob
