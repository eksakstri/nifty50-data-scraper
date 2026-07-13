import asyncio
import json
import os
from datetime import datetime
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=30000
)

client.admin.command("ping")

print("MongoDB Atlas connected")

DB_NAME = "nifty50"

BASE_FOLDER = "../nifty50documents"


client = MongoClient(MONGO_URI)

db = client[DB_NAME]

pdf_collection = db["pdf_documents"]


async def download_pdf(request, pdf_url, save_path):

    try:

        response = await request.get(pdf_url)

        if not response.ok:
            print(
                f"Download failed: "
                f"{response.status} "
                f"{pdf_url}"
            )
            return False

        content = await response.body()

        with open(save_path, "wb") as f:
            f.write(content)

        return True

    except Exception as e:

        print(
            f"PDF download error: {e}"
        )

        return False


async def process_symbol(
    request,
    symbol,
    company_url
):

    print(f"\nProcessing {symbol}")

    api_url = (
        "https://www.nseindia.com/api/NextApi/"
        "apiClient/GetQuoteApi"
        f"?functionName=getCorporateAnnouncement"
        f"&symbol={symbol}"
        f"&marketApiType=equities"
        f"&noOfRecords=50"
    )

    try:

        response = await request.get(api_url)

        if not response.ok:

            print(
                f"{symbol}: "
                f"API failed "
                f"{response.status}"
            )

            return

        data = await response.json()

    except Exception as e:

        print(
            f"{symbol}: "
            f"{e}"
        )

        return

    announcements = []

    if isinstance(data, list):

        announcements = data

    elif isinstance(data, dict):

        if "data" in data:
            announcements = data["data"]

    print(
        f"{symbol}: "
        f"{len(announcements)} announcements"
    )

    company_folder = os.path.join(
        BASE_FOLDER,
        symbol,
        "pdfs"
    )

    os.makedirs(
        company_folder,
        exist_ok=True
    )

    for item in announcements:

        pdf_url = item.get("attchmntFile")

        if not pdf_url:
            continue

        existing = pdf_collection.find_one(
            {
                "pdf_url": pdf_url
            }
        )

        if existing:

            print(
                f"Already exists:"
                f" {symbol}"
            )

            continue

        filename = os.path.basename(
            urlparse(pdf_url).path
        )

        local_path = os.path.join(
            company_folder,
            filename
        )

        success = await download_pdf(
            request,
            pdf_url,
            local_path
        )

        if not success:
            continue

        document = {

            "symbol":
                symbol,

            "company_url":
                company_url,

            "subject":
                item.get("subject"),

            "announcement_text":
                item.get("attchmntText"),

            "announcement_date":
                item.get("date")
                or item.get(
                    "broadcastDateTime"
                ),

            "file_size":
                item.get("fileSize"),

            "pdf_url":
                pdf_url,

            "local_path":
                local_path,

            "status":
                "downloaded",

            "download_time":
                datetime.utcnow()
        }

        pdf_collection.insert_one(
            document
        )

        print(
            f"Downloaded:"
            f" {filename}"
        )


async def main():

    with open(
        "nifty50_companies.json",
        "r",
        encoding="utf-8"
    ) as f:

        companies = json.load(f)

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False
        )

        context = await browser.new_context()

        page = await context.new_page()

        print(
            "Opening NSE homepage..."
        )

        await page.goto(
            "https://www.nseindia.com",
            wait_until="networkidle",
            timeout=120000
        )

        await page.wait_for_timeout(
            5000
        )

        request = context.request

        for company in companies:

            try:

                await process_symbol(
                    request,
                    company["symbol"],
                    company["url"]
                )

            except Exception as e:

                print(
                    company["symbol"],
                    e
                )

        await browser.close()

    print("\nDone")


if __name__ == "__main__":
    asyncio.run(main())