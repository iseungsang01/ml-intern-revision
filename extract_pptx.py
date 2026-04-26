import zipfile
import xml.etree.ElementTree as ET

try:
    with zipfile.ZipFile(r"C:\Users\lss\Desktop\졸논 정리.pptx") as z:
        slides = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
        # Sort slides by number to maintain order
        slides.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split('/')[-1]))))
        for slide in slides:
            print(f"\n[{slide}]")
            xml_content = z.read(slide)
            tree = ET.fromstring(xml_content)
            ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
            for t in tree.findall('.//a:t', ns):
                if t.text:
                    print(t.text)
except Exception as e:
    print(f"Error: {e}")
