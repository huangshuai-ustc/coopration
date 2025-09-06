import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm

xml_file = 'ok.xml'
csv_file = 'ok.csv'


def get_text(element, default=''):
    """提取 style 标签中的文本"""
    if element is not None:
        style = element.find('style')
        return style.text.strip() if style is not None and style.text else default
    return default


def get_multiple_texts(elements):
    """提取多个元素中的 style 文本并用分号连接"""
    return '; '.join([
        style.text.strip()
        for el in elements
        for style in el.findall('style')
        if style.text
    ])


tree = ET.parse(xml_file)
root = tree.getroot()

records = root.findall('.//record')

with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'Title', 'Authors', 'Journal', 'Year', 'Volume', 'Issue',
        'Pages', 'DOI', 'Abstract', 'Keywords'
    ])
    writer.writeheader()

    for record in tqdm(records):
        title = get_text(record.find('./titles/title'))
        authors = get_multiple_texts(record.findall('./contributors/authors/author'))
        journal = get_text(record.find('./periodical/full-title'))
        year = get_text(record.find('./dates/year'))
        volume = get_text(record.find('./volume'))
        issue = get_text(record.find('./number'))
        pages = get_text(record.find('./pages'))
        doi = get_text(record.find('./electronic-resource-num'))
        abstract = get_text(record.find('./abstract'))
        keywords = get_multiple_texts(record.findall('./keywords/keyword'))

        writer.writerow({
            'Title': title,
            'Authors': authors,
            'Journal': journal,
            'Year': year,
            'Volume': volume,
            'Issue': issue,
            'Pages': pages,
            'DOI': doi,
            'Abstract': abstract,
            'Keywords': keywords
        })

print(f"✅ 成功导出 {len(records)} 条文献到 {csv_file}")
