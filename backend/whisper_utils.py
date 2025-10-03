import httpx, os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def run_whisper(file_bytes: bytes, filename="audio.wav"):
    url = 'https://api.openai.com/v1/audio/transcriptions'
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
    files = {'file': (filename, file_bytes), 'model': (None, 'whisper-1')}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, files=files)
        r.raise_for_status()
        return r.json()
