"""Check native HTML images and actual score JavaScript offline.

Qt's JS engine executes the page script against a minimal DOM adapter. This
checks score/reload/reset/export logic, not browser layout. PIL draws a phone
preview; the full 1x/3x texture audit is compare.png.
"""
import base64
from html.parser import HTMLParser
from io import BytesIO
import json
import re
from PIL import Image, ImageDraw
from .common import HOME, font, write_json


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.cards, self.card = [], [], None
        self.heading = False
        self.blind_depth, self.depth = 0, 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article':
            self.card = dict(label='', images=[])
            self.cards.append(self.card)
        if tag == 'h2': self.heading = True
        if tag == 'div':
            self.depth += 1
            if attrs.get('class') == 'blind': self.blind_depth = self.depth
        if tag == 'img':
            rgb = Image.open(BytesIO(base64.b64decode(attrs['src'].split(',')[1]))).convert('RGB')
            w, h = int(attrs['width']), int(attrs['height'])
            assert w/rgb.width == h/rgb.height and w/rgb.width in (1, 3)
            self.images.append(rgb)
            if self.blind_depth: self.card['images'].append(rgb)

    def handle_endtag(self, tag):
        if tag == 'h2': self.heading = False
        if tag == 'div':
            if self.depth == self.blind_depth: self.blind_depth = 0
            self.depth -= 1

    def handle_data(self, data):
        if self.heading and self.card is not None: self.card['label'] += data


DOM = r'''
const __elements = {};
let __storage = __INITIAL__;
let __blob = null, __download = '';
function element(data) {return {dataset:data||{},disabled:false,textContent:'',classList:{toggle(){}},
  addEventListener(type,fn){this.clickHandler=fn},click(){this.clickHandler()},remove(){}}}
const __buttons = Array.from({length:40},(_,i)=>element({trial:String(Math.floor(i/2)),side:String(i%2)}));
const document = {
 querySelector(s){if(!__elements[s])__elements[s]=element();return __elements[s]},
 querySelectorAll(s){if(s==='.pick')return __buttons;const m=s.match(/data-trial="(\d+)"/);return m?__buttons.filter(b=>b.dataset.trial===m[1]):[]},
 createElement(){return {href:'',download:'',click(){__download=this.download},remove(){}}},body:{appendChild(){}}};
const localStorage={getItem(k){return __storage[k]||null},setItem(k,v){__storage[k]=v}};
function Blob(items,opts){this.items=items;__blob=items.join('')}
const URL={createObjectURL(){return 'blob:test'},revokeObjectURL(){}};
function setTimeout(fn){fn()}
'''


def javascript_audit(script):
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtQml import QJSEngine
    app = QCoreApplication.instance() or QCoreApplication([])
    engine = QJSEngine()
    def run(code):
        value = engine.evaluate(code)
        if value.isError(): raise AssertionError(value.toString())
        return value
    run(DOM.replace('__INITIAL__', '{}').replace('const ', 'var ').replace('let ', 'var '))
    run(script)
    run('__buttons[0].click()')
    state = run('JSON.stringify(__storage)').toString()
    assert '1 answered' in run("document.querySelector('#score').textContent").toString()
    run('__buttons[1].click()')
    run("document.querySelector('#export').click()")
    exported = json.loads(run('__blob').toString())
    assert exported['answered'] == 1 and len(exported['results']) == 1
    assert run('__download').toString() == 'lithology-blind-results.json'
    engine = QJSEngine()
    run(DOM.replace('__INITIAL__', state).replace('const ', 'var ').replace('let ', 'var '))
    run(script)
    assert '1 answered' in run("document.querySelector('#score').textContent").toString()
    run("document.querySelector('#reset').click()")
    assert '0 answered' in run("document.querySelector('#score').textContent").toString()
    return dict(click=True, double_click_guard=True, reload=True, reset=True, export=True)


def preview(card, width):
    canvas = Image.new('RGB', (width, 844), '#f4f5f2')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 245), fill='white', outline='#c9d0ca')
    draw.text((20, 20), 'Lithology: real or', fill='#172321', font=font(27))
    draw.text((20, 56), 'generated?', fill='#172321', font=font(27))
    draw.text((20, 112), '20 patterns across 7 families.', fill='#172321', font=font(16))
    draw.text((20, 146), 'Choose the real image in each pair.', fill='#172321', font=font(16))
    draw.text((20, 204), '0 correct / 0 answered', fill='#172321', font=font(18))
    draw.rectangle((12, 270, width-12, 780), fill='white', outline='#c9d0ca')
    draw.text((28, 290), card['label'], fill='#172321', font=font(23))
    draw.text((28, 340), 'Choose the real image.', fill='#172321', font=font(16))
    x = 28
    for i, rgb in enumerate(card['images']):
        draw.rectangle((x, 385, x+rgb.width+24, 440+rgb.height), fill='#f7f8f5', outline='#68786e', width=2)
        draw.text((x+12, 395), chr(65+i), fill='#172321', font=font(18))
        canvas.paste(rgb, (x+12, 430)); x += rgb.width+42
    draw.text((28, 580), 'Unanswered', fill='#172321', font=font(17))
    draw.text((28, 640), 'Inspect samples, legend, contacts', fill='#172321', font=font(16))
    draw.text((28, 665), 'and statistics', fill='#172321', font=font(16))
    return canvas


def main():
    text = (HOME/'out/compare.html').read_text(encoding='utf-8')
    page = Page()
    page.feed(text)
    for card in page.cards:
        assert len(card['images']) == 2 and card['images'][0].size == card['images'][1].size
    script = re.search(r'<script>(.*?)</script>', text, re.S).group(1)
    audit = javascript_audit(script)
    for width, name in ((1100, 'desktop'), (390, 'phone')):
        preview(page.cards[0], width).save(HOME/f'out/compare_{name}_pil.png')
    result = dict(trials=len(page.cards), embedded_images=len(page.images), native_aspect=True,
                  javascript=audit, preview='PIL approximation; browser layout not tested')
    write_json(HOME/'out/compare_audit.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__': main()
