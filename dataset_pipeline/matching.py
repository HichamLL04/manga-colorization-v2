from pathlib import Path

from .common import PagePair, iter_images, normalize_token, slug


def parse_page_identity(root, path):
    relative = Path(path).relative_to(root)
    parts = list(relative.parts)
    title = slug(parts[0]) if len(parts) >= 3 else 'unknown'
    chapter = slug(parts[1]) if len(parts) >= 3 else 'unknown'
    page = normalize_token(parts[-1])
    pair_id = '{}_{}_{}'.format(title, chapter, page)
    return pair_id, title, chapter, page


def index_pages(root):
    index = {}
    for path in iter_images(root):
        pair_id, title, chapter, page = parse_page_identity(root, path)
        index[pair_id] = {
            'path': path,
            'title': title,
            'chapter': chapter,
            'page': page,
        }
    return index


def match_pages(bw_root, color_root):
    bw_index = index_pages(Path(bw_root))
    color_index = index_pages(Path(color_root))
    pairs = []

    for pair_id in sorted(set(bw_index) & set(color_index)):
        bw_item = bw_index[pair_id]
        color_item = color_index[pair_id]
        pairs.append(
            PagePair(
                pair_id=pair_id,
                title=bw_item['title'],
                chapter=bw_item['chapter'],
                page=bw_item['page'],
                bw_path=bw_item['path'],
                color_path=color_item['path'],
            )
        )

    return pairs

