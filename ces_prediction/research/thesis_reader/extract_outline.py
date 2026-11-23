"""Read the author's HWP locally; retain private body text only in ignored data."""
import hashlib
import json
import re
import struct
import zlib
from pathlib import Path

import olefile

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / 'data/.thesis_reader'
OUT = ROOT / 'docs/paper/research_reader'


def paragraphs(path):
    with olefile.OleFileIO(path) as hwp:
        header = hwp.openstream('FileHeader').read()
        compressed = bool(struct.unpack_from('<I', header, 36)[0] & 1)
        result = []
        for stream in hwp.listdir():
            if stream[0] != 'BodyText':
                continue
            body = hwp.openstream(stream).read()
            if compressed:
                body = zlib.decompress(body, -15)
            offset = 0
            while offset + 4 <= len(body):
                record = struct.unpack_from('<I', body, offset)[0]
                tag, size = record & 0x3ff, record >> 20
                offset += 4
                if size == 0xfff:
                    size = struct.unpack_from('<I', body, offset)[0]
                    offset += 4
                payload = body[offset:offset+size]
                offset += size
                if tag == 67:
                    words = struct.unpack('<'+'H'*(len(payload)//2), payload[:len(payload)//2*2])
                    plain = []
                    pos = 0
                    while pos < len(words):
                        code = words[pos]
                        if code < 32:
                            if code in (0, 10, 13, 24, 30, 31):
                                if code in (10, 13, 30, 31):
                                    plain.append(' ')
                                pos += 1
                            else:
                                if code == 9:
                                    plain.append(' ')
                                pos += 8
                        else:
                            plain.append(chr(code)); pos += 1
                    text = ''.join(plain).strip()
                    if text:
                        result.append(text)
        return result


def main():
    path = Path.home() / 'Desktop/졸업논문_이승상_v1.hwp'
    if not path.exists():
        raise FileNotFoundError(path.name)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = paragraphs(path)
    (CACHE / 'hwp_body_private.txt').write_text('\n'.join(lines), encoding='utf8')
    # Only short heading-shaped paragraphs enter the reusable outline. No body,
    # author, institution, student identifier, or correspondence is transmitted.
    pattern = re.compile(r'^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]+[.．]?|[1-9][0-9]?[.．)]|[가-하][.．)]|[a-z][.．)])\s*')
    headings = [line for line in lines if len(line) < 140 and pattern.match(line)]
    record = {'source_filename': 'original_thesis_v1.hwp', 'source_name_redacted': True,
              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_format': 'HWP5 locally parsed paragraph records',
              'private_body_committed': False, 'headings': headings}
    (CACHE / 'source_receipt.json').write_text(json.dumps({'original_absolute_path':str(path),
        'source_sha256':record['source_sha256']}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (OUT / 'original_outline.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print('HWP parsed:', len(lines), 'paragraphs;', len(headings), 'heading candidates')


if __name__ == '__main__':
    main()
