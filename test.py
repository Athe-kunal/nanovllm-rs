import asyncio, httpx

prompts = ["The capital of France is", "The capital of Japan is", "2 + 2 ="]

async def generate(client, prompt):
    r = await client.post("http://localhost:8000/generate",
                           json={"prompt": prompt, "max_tokens": 1024})
    return r.json()

async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        results = await asyncio.gather(*(generate(client, p) for p in prompts))
    for prompt, result in zip(prompts, results):
        print(prompt, "->", result["text"])

asyncio.run(main())