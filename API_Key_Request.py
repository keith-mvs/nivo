
import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True


headers = {
  "Authorization": "Bearer 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",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "meta/llama-3.2-11b-vision-instruct",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 512,
  "temperature": 1.00,
  "top_p": 1.00,
  "frequency_penalty": 0.00,
  "presence_penalty": 0.00,
  "stream": stream
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
