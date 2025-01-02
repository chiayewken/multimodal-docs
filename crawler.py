"""
From https://www.manualslib.com/brand/, click all alphabet links in top bar
From https://www.manualslib.com/brand/A.html, click all brand links and number pages in bottom bar
From https://www.manualslib.com/brand/a-link/, click all model links which have number of pages in title (eg <div class="col-xs-9 col-sm-10 manuals-col"> <a href="/manual/704162/A-Link-Pa200av.html#product-PA200AVb" title="33 pages Ethernet Powerline Adapter">User Manual</a> </div>
Save the pdf page link in crawl_outputs.csv
"""
import json
import re
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from fire import Fire
from tqdm import tqdm


def get_letter_pages() -> List[str]:
    outputs = ["https://www.manualslib.com/brand/lit.html"]
    for i in range(10):
        outputs.append(f"https://www.manualslib.com/brand/{i}.html")
    for i in range(26):
        outputs.append(f"https://www.manualslib.com/brand/{chr(i + 65)}.html")
    return outputs


def get_soup(url):
    response = requests.get(url)
    return BeautifulSoup(response.content, "html.parser")


def get_brand_pages(url: str, min_subcategories: int = 2) -> List[dict]:
    # eg input: https://www.manualslib.com/brand/T.html
    # brands with more subcategories likely have more manuals
    soup = get_soup(url)
    page_links = soup.select("a.plink")

    # Extract the href attributes (links) from the found elements
    template = "https://www.manualslib.com{}"
    unique_links = sorted(set(template.format(link["href"]) for link in page_links))
    if not unique_links:
        unique_links = [url]

    outputs = []
    for link in unique_links:
        soup = get_soup(link)
        rows = soup.find_all("div", class_="row tabled")
        for row in rows:
            # Find the first anchor tag within the div with class 'col1'
            name = row.find("div", class_="col1").find("a").text
            brand_link = row.find("div", class_="col1").find("a")
            labels = sorted(row.find("div", class_="catel").stripped_strings)
            labels = [x for x in labels if len(x) > 1]
            url = f"https://www.manualslib.com{brand_link['href']}"
            outputs.append(dict(name=name, url=url, subcategories=labels))

    print(dict(url=url, unique_links=unique_links, brands=len(outputs)))
    return outputs


def get_manual_pages(url: str, min_pages: int = 100) -> List[str]:
    # eg input: https://www.manualslib.com/brand/hlins/
    soup = get_soup(url)
    manual_links = soup.find_all("a", href=re.compile(r"/manual/"))

    outputs = []
    seen = set()
    for link in manual_links:
        href = link.get("href")
        title = link.get("title")
        if href is None or title is None:
            continue

        page_match = re.search(r"(\d+)\s+pages", title)
        num_pages = int(page_match.group(1)) if page_match else 0
        if num_pages >= min_pages:
            key = (title, num_pages)
            if key in seen:
                continue
            seen.add(key)
            outputs.append(f"https://www.manualslib.com{href}")
            print(dict(title=title, num_pages=num_pages, link=outputs[-1]))

    return outputs


def save_brands(path_out: str = "data/crawl/brands.json"):
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, "w") as f:
        for letter_page in tqdm(get_letter_pages()):
            for info in get_brand_pages(letter_page):
                print(path_out, json.dumps(info))
                print(json.dumps(info), file=f)


"""
python crawler.py get_letter_pages
python crawler.py get_brand_pages https://www.manualslib.com/brand/lit.html
python crawler.py get_brand_pages https://www.manualslib.com/brand/A.html
python crawler.py get_manual_pages https://www.manualslib.com/brand/abqindustrial/ --min_pages 10
python crawler.py save_brands
"""


if __name__ == "__main__":
    Fire()
