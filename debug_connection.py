import httpx
import os
import asyncio

url = os.environ.get("AIRFLOW__CORE__INTERNAL_API_URL")
print(f"Testing connection to: {url}")

async def test():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{url}health")
            print(f"Response: {resp.status_code}")
            print(resp.json())
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(test())
