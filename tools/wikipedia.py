import argparse
import datetime
import json
import os

import requests


class Queue:
    def __init__(self, first_item, max_depth=0):
        self.processed = set()
        self.to_process = [first_item]
        self.depth = 0
        self.length_current_depth = 1
        self.length_next_depth = 0
        self.max_depth = max_depth

    def add(self, item):
        self.add_all([item])

    def add_all(self, items):
        for item in items:
            if item not in self.processed:
                self.to_process.append(item)
                self.processed.add(item)
                self.length_next_depth += 1

    def pop(self):
        if self.length_current_depth == 0:
            self.depth += 1
            self.length_current_depth = self.length_next_depth
            self.length_next_depth = 0
        self.length_current_depth -= 1
        return self.to_process.pop(0)

    def empty(self):
        return len(self.to_process) == 0 or (self.max_depth and self.depth > self.max_depth)


def get_categories(super_category, paging=None):
    params = {'action': 'query', 'list': 'categorymembers',
              'cmtitle': super_category,
              'format': 'json', 'cmlimit': 50}
    if paging:
        params['cmcontinue'] = paging
    response = requests.get(wiki_url, params=params)
    response_json = json.loads(response.text)
    titles = [item['title'] for item in response_json['query']['categorymembers']]

    if 'continue' in response_json and 'cmcontinue' in response_json['continue']:
        titles += get_categories(super_category, paging=response_json['continue']['cmcontinue'])

    return titles


def get_categories_recursive(super_category, max_depth=1):
    page_titles = set()
    queue = Queue(first_item=super_category, max_depth=max_depth)
    while not queue.empty():
        titles = get_categories(queue.pop())
        for title in titles:
            if title.startswith(category_prefix):
                queue.add(title)
            else:
                page_titles.add(title)
    return page_titles, queue.processed


def get_plain_text(page):
    params = {'action': 'query', 'prop': 'extracts', 'explaintext': '',
              'titles': page, 'format': 'json'}

    response = requests.get(wiki_url, params=params)
    response_json = json.loads(response.text)
    plaintext = list(response_json['query']['pages'].values())[0]['extract']
    return plaintext.replace('=', ' ').replace('\n', ' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get category from wikipedia.')

    parser.add_argument('dir', type=str, help='directory to write to')
    parser.add_argument('-l', '--lang', choices=['en', 'sv'], default='en', help='language')
    parser.add_argument('-i', '--ignore', type=str, nargs='?', help='file containing pages to ignore')
    parser.add_argument('-p', '--pages', type=str, nargs='?', help='file containing pages to read')
    parser.add_argument('-c', '--category', type=str, nargs='?', help='category')
    parser.add_argument('-d', '--depth', type=int, default=4, help='depth of category tree to read')

    args = parser.parse_args()

    assert args.category or args.pages

    wiki_url = 'https://{}.wikipedia.org/w/api.php'.format(args.lang)
    category_prefix = {'en': 'Category', 'sv': 'Kategori'}[args.lang] + ':'

    args.dir = os.path.join(args.dir, '{}-{}-{}'.format(args.lang,
                                                        args.category or os.path.split(args.pages)[1],
                                                        args.depth or 'continue'))
    print(datetime.datetime.now(), 'Will save to {}.'.format(args.dir))
    os.mkdir(args.dir)  # deliberately fail if exists

    print(datetime.datetime.now(), 'Getting categories and pages...')

    if not args.pages:
        pages, categories = get_categories_recursive(category_prefix + args.category, max_depth=4)
        print(datetime.datetime.now(), 'Found', len(categories), 'categories and', len(pages), 'pages.')

        print(datetime.datetime.now(), 'Writing categories and pages to file...')
        with open(os.path.join(args.dir, 'categories.txt'), 'w') as f:
            f.write('\n'.join(categories))
        with open(os.path.join(args.dir, 'pages.txt'), 'w') as f:
            f.write('\n'.join(pages))
    else:
        pages = []
        with open(os.path.join(args.pages), 'r') as f:
            for line in f:
                pages.append(line.strip())
        print(datetime.datetime.now(), 'Found', len(pages), 'pages.')

    processed_pages = set()
    if args.ignore:
        print(datetime.datetime.now(), 'Reading processed pages...')
        with open(args.ignore, 'r') as f:
            for line in f:
                processed_pages.add(line.strip())
        print(datetime.datetime.now(), 'Found', len(processed_pages), 'processed pages.')

    print(datetime.datetime.now(), 'Getting pages...')
    with open(os.path.join(args.dir, 'contents.txt'), 'w') as f:
        for i, page in enumerate(pages):
            if i % 100 == 0:
                print(datetime.datetime.now(), 'Getting page', i, page, '...')
            if page in processed_pages:
                continue
            f.write(page + '\n' + get_plain_text(page) + '\n\n')


