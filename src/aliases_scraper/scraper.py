import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


@dataclass
class CuisineAlias:
    cuisine_name: str
    lang_code: str
    lang: str


class WikipediaAnalyzer:

    def __init__(self):
        pass

    def __interlanguage_name(self, link: BeautifulSoup) -> CuisineAlias:
        """
        Get the cuisine's name in other language

        Args:
            link: BeautifulSoup, interlanguage li link

        Returns:
            CuisineAlias object
        """
        interlanguage_link_title = link.a.get("title")
        lang_code = link.a.get("lang")

        # in case wikipedia writes in "{cuisine name} - {language}" format
        if "–" in interlanguage_link_title: 
            components = interlanguage_link_title.split("–")
            cuisine_name = components[0].split("(")[0].strip()  # sometimes Wikipedia writes as XXX (food)
            language_name = components[1].strip()
        # in the case wikipedia writes in "{cuisine name} ({language})" format
        else:  
            components = interlanguage_link_title.split("(")
            cuisine_name = components[0].strip()
            language_name = "?"

        return CuisineAlias(cuisine_name, lang_code, language_name)

    def get_cuisine_aliases(self, soup: BeautifulSoup) -> list[CuisineAlias]:
        """
        Get cuisine alias names in other language, given a cuisine_url

        Args:
            cuisine_url: str, wikipedia cuisine url, e.g., 'https://en.wikipedia.org/wiki/Rawon'

        Returns:
            list[CuisineAlias]
        """
        # process html
        interlanguage_links = set(soup.find_all("li", class_="interlanguage-link"))
        all_aliases = []
        for link in interlanguage_links:
            cuisine_alias = self.__interlanguage_name(link)
            all_aliases.append(cuisine_alias)

        return all_aliases

    async def get_html(self, idx: int, session: aiohttp.ClientSession, cuisine_url: str) -> BeautifulSoup:
        """
        Async request to get html result

        Args:
            idx: int
            session: aiohttp.ClientSession
            cuisine_url: str

        Returns:
            BeautifulSoup object

        """
        async with session.get(cuisine_url) as resp:
            # Requesting the page
            response = await resp.text()

            # process html
            soup = BeautifulSoup(response, "html.parser")

            return idx, soup


async def process_inputs(df: pd.DataFrame, analyzer: WikipediaAnalyzer, cutoff: int | None = None):
    """
    Args:
        df: pd.DataFrame, input file in DataFrame format
        analyzer: WikipediaAnalyzer
        cutoff: int, only use the first cutoff-th data
            This is for trial run

    Returns:
        pd.DataFrame
    """

    def aliases_to_ordered_dict(aliases_entry: Tuple[int, list[CuisineAlias]]):
        d = {}
        aliases = aliases_entry[1]
        for alias in aliases:
            d[f"{alias.lang_code} - {alias.lang}"] = alias.cuisine_name
        d = OrderedDict(sorted(d.items()))
        return d

    urls = df.loc[:, "Wikipedia Link"].tolist()
    if cutoff is not None:
        urls = urls[:cutoff]

    async with aiohttp.ClientSession() as session:
        # coroutine tasks
        tasks = set()
        for idx, url in enumerate(urls):
            task = asyncio.create_task(analyzer.get_html(idx, session, url))
            tasks.add(task)

        # unpacking results
        unique_languages = {}
        all_aliases: list[Tuple[int, list[CuisineAlias]]] = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            idx, soup = await task
            aliases = analyzer.get_cuisine_aliases(soup)
            for alias in aliases:
                if alias.lang != "?":
                    unique_languages[alias.lang_code] = alias.lang
            all_aliases.append((idx, aliases))
        all_aliases.sort()

        # handling Wikipedia's inconsistent formats
        for _, aliases in all_aliases:
            for alias in aliases:
                if alias.lang == "?":
                    if alias.lang_code in unique_languages:
                        alias.lang = unique_languages[alias.lang_code]

        # output formatting
        unique_languages = sorted([f"{key} - {val}" for key, val in unique_languages.items()])  # type: ignore
        n = len(df) if cutoff is None else cutoff
        new_data = []
        for i in range(n):
            new_row = []
            info = aliases_to_ordered_dict(all_aliases[i])
            aliases_json_dump =  json.dumps(info, ensure_ascii=False)
            new_row.append(aliases_json_dump)
            for lang in unique_languages:
                new_row.append(info[lang] if lang in info else "")    
            new_data.append(new_row)
        
        # combine new data to old data
        column_names = ['json_aliases']
        column_names.extend(unique_languages)
        new_df = pd.DataFrame(data=new_data, columns=column_names)
        if cutoff is not None:
            df = df.loc[:cutoff]
        combined_df = pd.concat([df, new_df], axis=1)
        return combined_df


if __name__ == "__main__":
    # IOs
    input_filepath = "aliases_scraper/target.xlsx"
    output_filepath = "aliases_scraper/outputs.xlsx"

    # scraping
    df = pd.read_excel(input_filepath)
    analyzer = WikipediaAnalyzer()
    df = asyncio.run(process_inputs(df, analyzer, cutoff=None))

    # save to external file
    df.to_excel(output_filepath)
