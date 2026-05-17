from .common import load_json


def merge_captions(panel_id, caption_sources=None):
    captions = []
    for source in caption_sources or []:
        payload = load_json(source, default={})
        value = payload.get(panel_id)
        if isinstance(value, dict):
            value = value.get('caption') or value.get('text')
        if value:
            captions.append(str(value).strip())

    if captions:
        plain = ' '.join(dict.fromkeys(captions))
    else:
        plain = 'manga panel, clean line art, colored comic illustration'

    sd_prompt = '{}. high quality manga coloring, clean edges, consistent character colors'.format(plain)
    return {
        'plain': plain,
        'stable_diffusion_prompt': sd_prompt,
        'sources': captions,
    }

