import time
import xml.etree.ElementTree as ET
import re

ns = {'ns': 'http://www.mediawiki.org/xml/export-0.10/'}

context = ET.iterparse('nowiki-20240501-pages-articles.xml', events=('end',))

land_code_regex = re.compile(r'^:[a-z]{1,2}:', re.IGNORECASE)
land_code_simple = re.compile(r'^[a-z]{1,2}:')

all_links_with_colon = set()

invalid_prefix = [
    "kategori", "category",
    "fil", "file",
    "image", "bilde",
    "mal",
    "media"
    "bruker", "user",
    "mediawiki",
    "wikipedia",
    "wikidata",
    "sak",
    "special",
    "Spesial",
    "wikt",
    "wikisource",
    "wikicities"
    "iarchive",
    "www",
    "commons",
    "meta",
    "portal",
    "modul",
    "D",
    "wiktionary",
    "ShareMap",
    "hjelp",
    "doi",
    "lokalhistoriewiki",
    "stq",
    "roa-rup"
]

invalid_start = [
    "---",
    "===="
]

invalid_links = [
    "#ifexpr",
    r"{{",
    r"}}",
    r"\n",
    "https://",
    "http://"
]

invalid_prefix_pattern = r"(?::)?(" + "|".join(re.escape(prefix) for prefix in invalid_prefix) + r"):"
invalid_prefix_regex = re.compile(invalid_prefix_pattern, re.IGNORECASE)

invalid_start_pattern = r"^(" + "|".join(re.escape(start) for start in invalid_start) + r")"
invalid_start_regex = re.compile(invalid_start_pattern)

invalid_links_pattern = r"(" + "|".join(re.escape(link) for link in invalid_links) + r")"
invalid_links_regex = re.compile(invalid_links_pattern)

def is_invalid_word(word):
    if invalid_prefix_regex.match(word):
        return True

    if invalid_start_regex.match(word):
        return True

    if invalid_links_regex.search(word):
        return True

    if bool(land_code_regex.match(word)):
        return True

    if bool(land_code_simple.match(word)):
        return True

    return False


def find_all_colons(words):
    for word in words:
        if ':' in word:
            all_links_with_colon.add(word)


def find_edges(article: str):
    pattern = r'\[\[([^\[\]]+?)(?:\|[^\[\]]*?)?\]\]'
    matches = re.findall(pattern, article)
    return list(filter(lambda x: not is_invalid_word(x), matches))


def generate_graph_file_b():
    start_time = time.perf_counter()
    filename = 'wikipedia_graph.txt'

    with open(filename, 'w', encoding='utf-8') as _:
        pass

    vertices = []
    edges = []

    n = 0
    for _, elem in context:
        if elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
            title = elem.find('ns:title', ns).text

            if is_invalid_word(title):
                continue

            revision_elem = elem.find('ns:revision', ns)
            text_elem = revision_elem.find('ns:text', ns).text

            try:
                e = find_edges(text_elem)
            except:
                continue

            find_all_colons(e)

            vertices.append(title)
            edges.append(e)

            n += 1

            if len(vertices) > 5000:
                write_to_file(vertices, edges, filename)
                vertices.clear()
                edges.clear()

            if n % 10000 == 0:
                print('.', end='', flush=True)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nProcessed {n} number of articles in {elapsed_time:.2f} seconds")


def write_to_file(vertices: list, edges: list, filename: str):
    lines = []

    for i in range(len(vertices)):
        title = vertices[i]
        edges_list = edges[i]

        lines.append(f"{title}$${"$$".join(edges_list)}")

    with open(filename, 'a', encoding='utf-8') as file:
        file.write("\n".join(lines))


if __name__ == '__main__':
    generate_graph_file_b()

    print(len(all_links_with_colon))
    print("-" * 20)
    print(all_links_with_colon)
    print("-" * 20)

    print(len(all_links_with_colon))
