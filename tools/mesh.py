# mesh_files = ['/Users/xx/thesis/mesh/c2018.txt', '/Users/xx/thesis/mesh/d2018.txt', '/Users/xx/thesis/mesh/q2018.txt']
import codecs
import json
from collections import defaultdict

import requests
from lxml.html import fromstring


id_to_term = {}
term_to_id = {}
trec_mesh_terms = []


def parse_mesh_file(file='/Users/xx/thesis/mesh/cdq1997.txt'):
    def get_value(l):
        _, value = l.split('=')
        return value.strip()

    with codecs.open(file, encoding='ISO-8859-1') as f:
        for line in f:
            line = line.strip()
            if line == '*NEWRECORD':
                uid = ''
                term = ''
            elif line.startswith('MH =') or line.startswith('SH =') or line.startswith('NM ='):
                term = get_value(line).lower()
            elif line.startswith('UI ='):
                uid = get_value(line)
                term_to_id[term] = uid
                id_to_term[uid] = term


def parse_trec_mesh(file='/Users/xx/thesis/ir-datasets/ohsu-trec/trec9-train/query.mesh.1-4904'):
    with open(file) as f:
        for line in f:
            if line.startswith('<title>'):
                term = line[len('<title>'):].strip().lower()
                trec_mesh_terms.append(term)


def df_ohsu_trec(file='/Users/xx/thesis/ir-standard/ohsu-trec/ohsu-trec-documents.txt'):
    dfs = defaultdict(int)
    with open(file) as f:
        for line in f:
            if not line or line.startswith('.'):
                continue
            words = line.split()
            for word in words:
                dfs[word] += 1
    return sorted(dfs, key=dfs.get, reverse=True)


def generate_dictionary(words):
    for word in words:
        if word in term_to_id:
            swedish_name = get_swedish_name(term_to_id[word])
            if swedish_name and ' ' not in swedish_name:
                print(word, swedish_name.lower())


umls_ticket_generating_ticket_uri = 'https://utslogin.nlm.nih.gov'
umls_auth_endpoint = '/cas/v1/api-key'
umls_ticket_uri = 'http://umlsks.nlm.nih.gov'
umls_api_uri = 'https://uts-ws.nlm.nih.gov'
umls_mesh_swe_endpoint = '/rest/content/current/source/MSHSWE/'


def get_umls_ticket_generating_ticket(api_key):
    params = {'apikey': api_key}
    header = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
    response = requests.post(umls_ticket_generating_ticket_uri + umls_auth_endpoint, data=params, headers=header)
    response_text = fromstring(response.text)
    return response_text.xpath('//form/@action')[0]


def get_umls_ticket(ticket_generating_ticket_uri):
    params = {'service': umls_ticket_uri}
    header = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
    response = requests.post(ticket_generating_ticket_uri, data=params, headers=header)
    return response.text


def get_swedish_name(uid):
    try:
        params = {'ticket': get_umls_ticket(ticket_generating_ticket)}
        response = requests.get(umls_api_uri + umls_mesh_swe_endpoint + uid, params=params)
        response_json = json.loads(response.text)['result']
        return response_json['name']
    except:
        return None


parse_mesh_file()
# parse_trec_mesh()

ticket_generating_ticket = get_umls_ticket_generating_ticket('<INSERT_API_KEY_HERE>')

generate_dictionary(df_ohsu_trec())

