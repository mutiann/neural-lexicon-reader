import requests
import os
import random
import urllib
import time
import bs4

base_path = r"D:\data\internal\lexicon"
cache_path = r"D:\cache"

r = requests.session()

def get_headers():
    user_agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36 Edg/87.0.664.66"] + \
     ['Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
                    'Mozilla/5.0 (compatible; adidxbot/2.0; +http://www.bing.com/bingbot.htm)',
                    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534+ (KHTML, like Gecko) BingPreview/1.0b']
    headers = {'User-Agent': random.choice(user_agents)}
    return headers


def get_page(key, url, cache_dir, encoding='utf-8'):
    if os.path.exists(os.path.join(cache_dir, key + ".html")):
        page_text = open(os.path.join(cache_dir, key + ".html"), encoding="utf-8").read()
    else:
        # url = urllib.parse.quote_plus(url, safe='')
        page = r.get(url, headers=get_headers())
        if page.status_code == 403:
            time.sleep(5)
            raise ValueError("403")
        page_text = page.text
        if encoding != 'utf-8':
            page.encoding = 'iso-8859-1'
            page_text = page.text.encode('iso-8859-1').decode(encoding, 'backslashreplace')
        open(os.path.join(cache_dir, key + ".html"), 'w', encoding="utf-8").write(page_text)
    return page_text

def get_text(tag):
    _inline_elements = {"a", "span", "em", "strong", "u", "i", "font", "mark", "label", "s", "sub", "sup", "tt", "bdo",
                        "button", "cite", "del", "b", "a", "font"}
    results = []

    def _get_text(tag):
        for child in tag.children:
            if isinstance(child, bs4.Tag):
                # if the tag is a block type tag then yield new lines before & after
                is_block_element = child.name not in _inline_elements
                if is_block_element:
                    results.append("\n")
                if child.name == "br":
                    results.append("\n")
                else:
                    _get_text(child)
                if is_block_element:
                    results.append("\n")
            else:
                results.append(child.string)
    _get_text(tag)
    return "".join(results)