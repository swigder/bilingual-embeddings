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


def download_wikipedia(args):
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


def merge(args):
    all_pages = []
    with open(os.path.join(args.to_dir, 'pages.txt'), 'r') as f:
        for line in f:
            all_pages.append(line.strip())

    processed_pages = []
    with open(os.path.join(args.to_dir, 'contents.txt'), 'r') as f:
        previous_is_blank = True
        for line in f:
            line = line.strip()
            if previous_is_blank and line:
                processed_pages.append(line)
            previous_is_blank = not bool(line)

    with open(os.path.join(args.to_dir, 'contents.txt'), 'a') as to_file:
        with open(os.path.join(args.from_dir, 'contents.txt'), 'r') as from_file:
            previous_is_blank = True
            add_current_line = False
            for line in from_file:
                line = line.strip()
                if previous_is_blank and line in all_pages and line not in processed_pages:
                    to_file.write(line + '\n')
                    add_current_line = True
                elif add_current_line:
                    if line:
                        to_file.write(line + '\n')
                    to_file.write('\n')
                    add_current_line = False
                previous_is_blank = not bool(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get category from wikipedia.')

    subparsers = parser.add_subparsers()

    parser_download = subparsers.add_parser('dl')
    parser_download.add_argument('dir', type=str, help='directory to write to')
    parser_download.add_argument('-l', '--lang', choices=['en', 'sv'], default='en', help='language')
    parser_download.add_argument('-i', '--ignore', type=str, nargs='?', help='file containing pages to ignore')
    parser_download.add_argument('-p', '--pages', type=str, nargs='?', help='file containing pages to read')
    parser_download.add_argument('-c', '--category', type=str, nargs='?', help='category')
    parser_download.add_argument('-d', '--depth', type=int, default=4, help='depth of category tree to read')
    parser_download.set_defaults(func=download_wikipedia)

    parser_merge = subparsers.add_parser('merge')
    parser_merge.add_argument('to_dir', type=str, help='directory to merge into')
    parser_merge.add_argument('from_dir', type=str, help='directory to merge from')
    parser_merge.set_defaults(func=merge)

    args = parser.parse_args()

    args.func(args)


