import requests
import json
import time
import os

def direct_test():
    api_key = os.getenv("GOOGLE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    # We use the direct REST endpoint for the most "naked" test possible
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": "Say 'The API key is working perfectly!' and nothing else."}]
        }]
    }
    
    print("--- Starting Direct REST API Test ---")
    print("Sending direct request to Google Gemini...")
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        try:
            text = data['candidates'][0]['content']['parts'][0]['text']
            print(f"\n[SUCCESS] Gemini Response: {text.strip()}")
        except Exception:
            print("\n[SUCCESS] API Response received (Key is valid), but format was unexpected.")
            print(json.dumps(data, indent=2))
    elif response.status_code == 429:
        print("\n[VERIFIED] The API key IS working and authenticated!")
        print("However, you have hit your Free Tier limit for the next 60 seconds.")
        print("Wait 1 minute, then this script will work.")
    else:
        print(f"\n[FAILED] Status {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    direct_test()
