import requests
import base64
import json

# ============================================================
# CONFIG
# ============================================================

API_KEY = "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"

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

# ONLY MODEL CHANGED
MODEL = "z-ai/glm-5.3-flash"

LOCAL_IMAGE = r"C:\Users\Jayachandran\ProjectsAndDocs\doc.pdf"


# ============================================================
# API KEY
# ============================================================

if not API_KEY:
    raise RuntimeError("Please put your NVIDIA API key in API_KEY")


# ============================================================
# IMAGE -> BASE64
# ============================================================

with open(LOCAL_IMAGE, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

image_url = f"data:image/jpeg;base64,{image_b64}"


# ============================================================
# HEADERS
# ============================================================

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}


# ============================================================
# INVOICE PROMPT
# ============================================================

system_prompt = """
You are a precise document data extraction engine."""


# ============================================================
# PAYLOAD
# ============================================================

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ],
    "max_tokens": 16384,
    "temperature": 0.5,
    "top_p": 1,
    "stream": False
}


# ============================================================
# REQUEST
# ============================================================

response = requests.post(
    invoke_url,
    headers=headers,
    json=payload,
    timeout=120
)


# ============================================================
# RESULT
# ============================================================

print("Status:", response.status_code)

response.raise_for_status()

result = response.json()

print("\n================ RAW RESPONSE ================\n")
print(json.dumps(result, indent=2))


# ============================================================
# EXTRACT MODEL CONTENT
# ============================================================

content = result["choices"][0]["message"]["content"]

print("\n================ INVOICE JSON ================\n")
print(content)


# ============================================================
# VALIDATE THAT MODEL RETURNED JSON
# ============================================================

try:
    invoice = json.loads(content)

    print("\n================ PARSED JSON ================\n")
    print(json.dumps(invoice, indent=2, ensure_ascii=False))

except json.JSONDecodeError:
    print("\nWARNING: Model did not return valid JSON.")