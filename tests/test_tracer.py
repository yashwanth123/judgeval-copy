
from judgeval.common.tracer import tracer


@tracer.observe(name="generate_movie_review", top_level=True)
def generate_movie_review(summary: str) -> str:
    # Analyze key elements
    plot_quality = analyze_plot(summary)
    engagement = analyze_engagement(summary)
    originality = analyze_originality(summary)
    
    # Generate final review
    review = compose_review(plot_quality, engagement, originality)
    return review

@tracer.observe(name="analyze_plot")
def analyze_plot(summary: str) -> dict:
    # Analyze plot elements like structure, pacing, coherence
    return {
        "structure": 8,  # 1-10 rating
        "pacing": 7,
        "coherence": 9,
        "notes": "Well structured plot with good pacing"
    }

@tracer.observe(name="analyze_engagement") 
def analyze_engagement(summary: str) -> dict:
    # Analyze how engaging/interesting the story seems
    return {
        "interest_level": 8,
        "emotional_impact": 7,
        "memorability": 8,
        "notes": "Engaging story with emotional resonance"
    }

@tracer.observe(name="analyze_originality")
def analyze_originality(summary: str) -> dict:
    # Analyze uniqueness and creativity
    return {
        "uniqueness": 6,
        "creativity": 7,
        "innovation": 5,
        "notes": "Some fresh elements but follows familiar patterns"
    }

@tracer.observe(name="compose_review")
def compose_review(plot: dict, engagement: dict, originality: dict) -> str:
    # Calculate overall score
    plot_score = sum([plot["structure"], plot["pacing"], plot["coherence"]]) / 3
    engagement_score = sum([engagement["interest_level"], 
                            engagement["emotional_impact"],
                            engagement["memorability"]]) / 3
    originality_score = sum([originality["uniqueness"],
                            originality["creativity"], 
                            originality["innovation"]]) / 3
    
    overall_score = (plot_score + engagement_score + originality_score) / 3
    
    # Generate review text
    review = f"""Movie Review:
Plot: {plot['notes']} ({plot_score:.1f}/10)
Engagement: {engagement['notes']} ({engagement_score:.1f}/10) 
Originality: {originality['notes']} ({originality_score:.1f}/10)

Overall Score: {overall_score:.1f}/10
"""
    return review

# Test the workflow
summary = """
A brilliant mathematician discovers a pattern that could predict global catastrophes. 
As she races to convince authorities of the impending doom, she must confront her own 
past traumas and decide whether to trust the pattern or her instincts. The fate of 
millions hangs in the balance as time runs out.
"""

result = generate_movie_review(summary)
result = result.result

print(type(result))
assert isinstance(result, str)
# assert "Movie Review:" in result
# assert "Overall Score:" in result

# Print the trace
# result.print_trace()
