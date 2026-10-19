import requests
from bs4 import BeautifulSoup
import urllib.parse
import yaml
from pathlib import Path


#Needs refinment

def scrape_incar_tags():
    """
    Scrapes all INCAR tag page titles from the VASP Wiki,
    following 'next page' links until none remain.
    Returns a sorted list of unique tags with spaces replaced by underscores.
    """
    base_url = "https://www.vasp.at"
    start_path = "/wiki/index.php?title=Category:INCAR_tag"  # changed here
    current_url = urllib.parse.urljoin(base_url, start_path)

    tags = set()
    while current_url:
        response = requests.get(current_url, headers={"User-Agent": "Mozilla/5.0"})
        if not response.ok:
            raise RuntimeError(f"Failed to retrieve: {current_url}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        tags.update(extract_incar_tags(soup))
        current_url = get_next_page_url(soup, base_url)

    return sorted(tags)


def extract_incar_tags(soup:BeautifulSoup):
    """Extracts and formats tag names from the current soup."""
    tags = set()
    for div in soup.select("div.mw-category"):
        for link in div.find_all("a"):
            tag = link.get_text().strip()
            if tag and not tag.startswith("("):
                tags.add(tag.replace(" ", "_"))
    return tags


def get_next_page_url(soup:BeautifulSoup, base_url:str):
    """Finds the URL of the next page if available."""
    nav_links = soup.select("#mw-pages a")
    for link in nav_links:
        text = link.get_text().lower()
        if "next page" in text or "next 200" in text:
            return urllib.parse.urljoin(base_url, link.get("href"))
    return None


def write_valid_incar_tags_in_txt(output_file:Path) -> None:
    tags = scrape_incar_tags()
    with open(output_file, "w") as f:
        for tag in tags:
            f.write(tag + "\n")


def write_valid_incar_tags_in_yaml(output_file:Path) -> None:
    tags = scrape_incar_tags()
    with open(output_file, "w") as f:
        yaml.dump({"valid_incar_tags":tags}, f, default_flow_style=False)

