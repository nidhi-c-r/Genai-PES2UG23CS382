from transformers import pipeline

# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

def analyze_review(review_text):
    """
    Takes a single review string and returns sentiment result.
    """
    result = sentiment_model(review_text)[0]
    label = result['label']
    score = round(result['score'], 4)

    return f"Sentiment: {label} (Confidence: {score})"


def analyze_multiple_reviews(review_list):
    """
    Takes a list of reviews and returns sentiment for each one.
    """
    results = sentiment_model(review_list)
    analyzed = []

    for review, result in zip(review_list, results):
        label = result['label']
        score = round(result['score'], 4)
        analyzed.append({
            "review": review,
            "sentiment": label,
            "confidence": score
        })

    return analyzed


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    # Single review
    review = "The product quality is amazing and exceeded my expectations!"
    print(analyze_review(review))

    # Multiple reviews
    reviews = [
        "I hated the product. Completely useless.",
        "Works really well, I love it!",
        "It's okay, nothing special.",
        "Terrible build quality. Not worth the price."
    ]

    results = analyze_multiple_reviews(reviews)

    print("\n--- Multiple Review Analysis ---")
    for res in results:
        print(f"\nReview: {res['review']}")
        print(f"Sentiment: {res['sentiment']} (Confidence: {res['confidence']})")
