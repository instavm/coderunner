import logging
import asyncio
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.DEBUG)

# Import version info using importlib.metadata
try:
    from importlib.metadata import version
    playwright_version = version('playwright')
except ImportError:
    playwright_version = "Unknown"

async def main():
    logging.info(f"Playwright version: {playwright_version}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect("ws://play20.local:3000/")
            print("Connected successfully")
    except Exception as e:
        logging.error(f"Connection failed: {e}")

# Run the async main method with asyncio
asyncio.run(main())