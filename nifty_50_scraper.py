import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


SAVE_DIR = Path("../embeddings")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


async def main():

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False
        )

        page = await browser.new_page()

        print("Opening NSE...")

        await page.goto(
            "https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%2050",
            wait_until="domcontentloaded",
            timeout=120000
        )

        # allow table to load
        await page.wait_for_timeout(10000)

        rows = await page.locator(
            "table tbody tr"
        ).all()

        print(f"Rows found: {len(rows)}")

        results = []

        for row in rows:

            try:

                cols = await row.locator("td").all_inner_texts()

                if len(cols) < 14:
                    continue

                item = {
                    "symbol": cols[0].strip(),
                    "open": cols[1].strip(),
                    "day_high": cols[2].strip(),
                    "day_low": cols[3].strip(),
                    "previous_close": cols[4].strip(),
                    "last_price": cols[5].strip(),
                    "index_close_price": cols[6].strip(),
                    "change": cols[7].strip(),
                    "percent_change": cols[8].strip(),
                    "volume": cols[9].strip(),
                    "traded_value_cr": cols[10].strip(),
                    "year_high": cols[11].strip(),
                    "year_low": cols[12].strip(),
                    "change_30d": cols[13].strip()
                }

                results.append(item)

            except Exception as e:
                print(e)

        snapshot = {
            "total_companies": len(results),
            "data": results
        }

        with open(
            SAVE_DIR / "../embeddings/nifty50_snapshot.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                snapshot,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Saved {len(results)} companies"
        )

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())