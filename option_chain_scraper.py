import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


DOWNLOAD_DIR = Path(
    "../embeddings"
)

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


async def main():

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False
        )

        context = await browser.new_context(
            accept_downloads=True
        )

        page = await context.new_page()

        print("Opening option chain...")

        await page.goto(
            "https://www.nseindia.com/option-chain",
            wait_until="networkidle",
            timeout=120000
        )

        await page.wait_for_timeout(
            5000
        )

        print(
            "Downloading CSV..."
        )

        async with page.expect_download() as dl:

            await page.locator(
                "#downloadOCTable"
            ).click()

        download = await dl.value

        file_path = (
            DOWNLOAD_DIR /
            "option_chain.csv"
        )

        await download.save_as(
            file_path
        )

        print(
            f"Saved -> {file_path}"
        )

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())