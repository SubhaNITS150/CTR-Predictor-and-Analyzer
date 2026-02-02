import os
import json
from google import genai

# The client will automatically look for GEMINI_API_KEY in your environment
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_llm_analysis(
    input_type: str,
    ctr_pct: float,
    features: dict,
    ad_text: str | None = None
) -> dict:

    prompt = f"""
You are an advertising performance expert.

Ad type: {input_type}
Predicted CTR: {ctr_pct:.2f}%

Extracted features:
{json.dumps(features, indent=2)}

Ad content:
{ad_text if ad_text else "N/A"}

TASK:
1. List 3–5 strengths (pros)
2. List 3–5 weaknesses (cons)
3. Give 4–6 actionable suggestions to improve CTR

OUTPUT STRICT JSON:
{{
  "pros": [],
  "cons": [],
  "suggestions": []
}}
"""

    try:
        # Using the new SDK's model and response format
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt,
            config={
                'response_mime_type': 'application/json' # Forces JSON output
            }
        )
        
        # The new SDK provides the parsed dict directly if you use response.parsed
        # or you can manually parse response.text
        return json.loads(response.text)

    except Exception as e:
        print("Gemini reasoning failed:", e)
        return {
            "pros": [],
            "cons": [],
            "suggestions": []
        }