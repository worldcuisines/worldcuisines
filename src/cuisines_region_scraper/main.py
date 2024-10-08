import pandas as pd
import requests
from bs4 import BeautifulSoup


def postprocessing(text: str) -> str:
    """
    Postprocessing cuisine type to remove "cuisine" prefix or suffix
    """
    if text is None:
        return text
    else:
        return text.lower().removesuffix("cuisines").removesuffix("cuisine").removeprefix("cuisine of")


def cuisine_region_scraper(url: str = "https://en.wikipedia.org/wiki/List_of_cuisines"):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    main_content_div = soup.find_all("div", {"class": "mw-content-ltr mw-parser-output"})[0]
    divs = main_content_div.find_all("div", {"class": ["mw-heading4", "mw-heading5", "div-col"]})

    region_cuisines = []
    curr_region = None
    curr_sub_region = None
    for div in divs:
        curr_classes = div.get("class")
        if "mw-heading4" in curr_classes:
            curr_region = div.h4.text
            curr_sub_region = None
        elif "mw-heading5" in curr_classes:
            curr_sub_region = div.h5.text
        elif "div-col" in curr_classes:
            if curr_classes is not None:
                cuisine_names = div.ul.find_all("li")
                for li_html in cuisine_names:
                    if len(li_html.find_all("ul")) > 0:
                        # actually, this is a sub_sub or sub_sub_sub title
                        cuisine_type = li_html.text.split("\n")[0]
                    else:
                        cuisine_type = li_html.text

                    region_cuisines.append(
                        [
                            postprocessing(curr_region),
                            postprocessing(curr_sub_region),
                            postprocessing(cuisine_type),
                        ]
                    )

    return region_cuisines


if __name__ == "__main__":
    results = cuisine_region_scraper()
    results_df = pd.DataFrame(results, columns=["h4 heading", "h5 heading", "li"])
    results_df.to_excel("outputs.xlsx")
