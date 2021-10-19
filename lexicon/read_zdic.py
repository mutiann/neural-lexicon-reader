from lexicon.lexicon_utils import *
from bs4 import BeautifulSoup
import re
import traceback
import json
import unicodedata
import tqdm

cache_dir = os.path.join(cache_path, 'zdic')
os.makedirs(cache_dir, exist_ok=True)
r = requests.session()
base_url = r"https://www.zdic.net/hans/"
lexicon = {'gloss': {}, 'oov': {}}
if os.path.exists(os.path.join(cache_dir, 'lexicon.json')):
    lexicon = json.load(open(os.path.join(cache_dir, 'lexicon.json'), encoding='utf-8'))

def query(word):
    if word in lexicon['gloss']:
        return lexicon['gloss'][word]
    if word in lexicon['oov']:
        return None
    try:
        result = query_direct(word)
    except:
        print(word)
        traceback.print_exc()
    if result is None:
        lexicon['oov'][word] = 1
    else:
        lexicon['gloss'][word] = result
    return result

def query_direct(word):
    if os.path.exists(os.path.join(cache_dir, word + ".html")):
        page_text = open(os.path.join(cache_dir, word + ".html"), encoding='utf-8').read()
    else:
        url = base_url + urllib.parse.quote_plus(word, safe='')
        page = r.get(url, headers=get_headers())
        if page.status_code == 403:
            time.sleep(5)
            raise ValueError("403")
        page_text = page.text
        open(os.path.join(cache_dir, word + ".html"), 'w', encoding='utf-8').write(page_text)
    soup = BeautifulSoup(page_text)
    section = soup.find('div', {'class': 'jbjs'})
    if section is None:
        return None
    definition = section.find('div', {'class': 'jnr'})
    if definition is None:
        definition = section.find('div', {'class': 'gnr'})
    if definition is None:
        return None
    t = definition.text.replace('“', '"').replace('”', '"')
    if (t.startswith('见"') or t.startswith('同"')) and len(definition.text) < 20:
        t = t.split('"')[1].strip()
        if t != word:
            return query(t)
        else:
            pass
    if definition.find(attrs={'class': 'dicpy'}) is None and definition.find('rt') is None:
        return None
    if definition.find(attrs={'class': 'dicpy'}) is not None and "," in definition.find(attrs={'class': 'dicpy'}).text:
        return None
    for class_name in ['copyright', 'enbox', 'ptr', 'encs', 'z_ts_2', 'gc_fy', 'gc_yy', 'gc_jy']:
        for tag in definition.find_all(attrs={'class': class_name}):
            tag.decompose()

    for p in definition.find_all("p"):
        if p.find(attrs={'class': 'smcs'}):
            p.decompose()

    for pinyin in definition.find_all(attrs={'class': 'dicpy'}):
        for p in pinyin.parent.previous_siblings:
            if isinstance(p, bs4.Tag):
                if p.find("strong") is not None and p.find("strong").text.replace("●", "").strip() != word:
                    p.decompose()
                break

    for ol in definition.find_all("ol"):
        lis = list(ol.find_all("li"))
        if len(lis) > 1:
            k = 0
            for i, item in enumerate(lis):
                item.replace_with("(%d) " % (i+1) + item.text)
    for example in definition.find_all(attrs={"class": "diczx1"}):
        t = example.text.replace('“', '"').replace('”', '"').strip('"').strip("'")
        example.replace_with('“' + t + '”')
    text = get_text(definition).replace("\n", ' ')
    text = text.replace("～", word).replace("◎", "")
    text = text.replace("● " + word, "● ").replace("∶", "：").\
        replace(") ：", ") ").replace(") ", ")").replace("( ", "(")
    text = text.replace("基本字义", "").replace("其它字义", "")
    text = ''.join([c for c in text if not (12549 <= ord(c) <= 12591) and not (
            12704 <= ord(c) <= 12735) and c not in "ˉˊˇˋ˙˪˫"]) # Remove bopomofo
    text = re.sub(r'\s+', ' ', text).strip()
    if text.startswith(word):
        text = "● " + text[len(word):].strip()
    text = "释义：" + word + ' ' + text
    return text

def fetch_gloss(vocab_path, update_path, out_path):
    vocab = json.load(open(vocab_path, encoding='utf-8'))
    subs = dict(json.load(open(update_path, encoding='utf-8'))) if update_path is not None else {}
    lengths = []
    for c in tqdm.tqdm(vocab):
        if not all([unicodedata.category(ch) == 'Lo' for ch in c]):
            continue
        if c in subs:
            gloss = subs[c]
        else:
            gloss = query(c)
        if gloss:
            lengths.append(len(gloss))
            if len(gloss) >= 200:
                print(gloss)

    json.dump(lexicon, open(os.path.join(cache_dir, 'lexicon.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump(lexicon, open(out_path, "w", encoding='utf-8'),
              ensure_ascii=False, indent=1)