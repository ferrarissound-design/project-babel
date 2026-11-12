"""正解転写を人が作る・直すためのローカルのアノテーション UI。

python -m babel annotate page.png --gt page.gt.json [--report glyph-report.json] [--port 8765]

- 127.0.0.1 だけで待ち受ける（外部に公開しない）。外部のライブラリ・API は使わない
- --gt のファイルがなければ、--report（babel vision の出力）から下書きを作る。下書きはすべて未確認・label なし
- 保存のたびに形式を検査する（groundtruth.validate）。不正なら保存しない
"""
from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import groundtruth as G

PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>BABEL 正解転写</title>
<style>
body{margin:0;font:13px system-ui,sans-serif;display:flex;height:100vh;background:#f4f4f2;color:#222}
#view{flex:1;overflow:auto;position:relative}#side{width:300px;padding:10px;overflow:auto;background:#fff;border-left:1px solid #ccc}
canvas{display:block;cursor:crosshair}label{display:block;margin:6px 0 2px}input[type=text],select{width:100%;box-sizing:border-box}
button{margin:3px 2px}.k{color:#666;font-size:12px}#msg{margin-top:8px;white-space:pre-wrap}
</style></head><body><div id="view"><canvas id="c"></canvas></div><div id="side">
<b>正解転写（babel-glyph-gt-v1）</b><div class="k">緑=確認済み 橙=未確認 赤=不確か 灰=文字以外 青=行</div>
<label>拡大 <input id="zoom" type="range" min="1" max="6" step="0.5" value="2"></label>
<div><button id="new">新しい枠 (n)</button><button id="del">削除 (Del)</button><button id="save">保存 (s)</button></div>
<div><button id="prev">前 (←)</button><button id="next">次 (→)</button><button id="vline">この行を確認済みに</button></div>
<div id="sel"><label>ID <span id="iid"></span></label>
<label>label（同じ形には同じ名前）<input id="label" type="text"></label>
<label>種類<select id="kind"><option>glyph</option><option>decoration</option><option>rule</option><option>noise</option></select></label>
<label>行<select id="line"></select></label>
<label>same_as（別名の統合先）<input id="same" type="text"></label>
<label><input id="ver" type="checkbox"> 人が確認した (v)</label>
<label><input id="unc" type="checkbox"> 不確か (u)</label>
<label><input id="outl" type="checkbox"> 中空（枠などの文字以外の要素。評価では縁だけを見る）</label>
<div class="k" id="prop"></div></div>
<div class="k">枠の中をドラッグで移動、右下の角でリサイズ。「新しい枠」の後にドラッグで追加。認識器の案（proposed）は参考で、正解ではない。</div>
<div id="msg"></div></div>
<script>
let gt=null,img=new Image(),sel=-1,mode=null,drag=null;const c=document.getElementById('c'),x=c.getContext('2d');
const $=id=>document.getElementById(id);const Z=()=>parseFloat($('zoom').value);
async function load(){gt=await (await fetch('/gt')).json();img.onload=()=>{draw();panel()};img.src='/image';}
function col(it){if(it.kind!=='glyph')return'#888';if(it.uncertain)return'#d22';return it.verified?'#2a2':'#e80';}
function draw(){const z=Z();c.width=img.width*z;c.height=img.height*z;x.imageSmoothingEnabled=false;x.drawImage(img,0,0,c.width,c.height);
 x.lineWidth=1;for(const ln of gt.lines){x.strokeStyle='#36c';x.setLineDash([3,3]);const b=ln.box;x.strokeRect(b[0]*z,b[1]*z,(b[2]-b[0])*z,(b[3]-b[1])*z);}
 x.setLineDash([]);gt.items.forEach((it,i)=>{const b=it.box;x.strokeStyle=col(it);x.lineWidth=i===sel?3:1;x.strokeRect(b[0]*z,b[1]*z,(b[2]-b[0])*z,(b[3]-b[1])*z);
 if(it.label){x.fillStyle=col(it);x.font='11px sans-serif';x.fillText(it.label,b[0]*z,b[1]*z-2);}});}
function panel(){const it=gt.items[sel];$('sel').style.opacity=it?1:.4;const ls=$('line');ls.innerHTML='<option value="">（なし）</option>'+gt.lines.map(l=>`<option>${l.id}</option>`).join('');
 if(!it)return;$('iid').textContent=it.id;$('label').value=it.label||'';$('kind').value=it.kind;ls.value=it.line||'';$('same').value=it.same_as||'';
 $('ver').checked=it.verified;$('unc').checked=it.uncertain;$('outl').checked=!!it.outline;$('prop').textContent=it.proposed?'認識器の案: '+JSON.stringify(it.proposed):'';}
function pos(e){const r=c.getBoundingClientRect(),z=Z();return[(e.clientX-r.left)/z,(e.clientY-r.top)/z];}
function hit(p){for(let i=gt.items.length-1;i>=0;i--){const b=gt.items[i].box;if(p[0]>=b[0]-1&&p[0]<=b[2]+1&&p[1]>=b[1]-1&&p[1]<=b[3]+1)return i;}return -1;}
c.onmousedown=e=>{const p=pos(e);if(mode==='new'){drag={t:'new',p0:p};return;}const i=hit(p);sel=i;panel();draw();if(i<0)return;const b=gt.items[i].box;
 drag={t:(Math.abs(p[0]-b[2])<3&&Math.abs(p[1]-b[3])<3)?'size':'move',p0:p,b0:b.slice()};};
c.onmousemove=e=>{if(!drag)return;const p=pos(e),dx=Math.round(p[0]-drag.p0[0]),dy=Math.round(p[1]-drag.p0[1]);
 if(drag.t==='move'){const b=drag.b0;gt.items[sel].box=[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy];}
 else if(drag.t==='size'){const b=drag.b0;gt.items[sel].box=[b[0],b[1],Math.max(b[0]+1,b[2]+dx),Math.max(b[1]+1,b[3]+dy)];}
 else{draw();const z=Z();x.strokeStyle='#e80';x.strokeRect(drag.p0[0]*z,drag.p0[1]*z,(p[0]-drag.p0[0])*z,(p[1]-drag.p0[1])*z);return;}draw();};
c.onmouseup=e=>{if(drag&&drag.t==='new'){const p=pos(e);const b=[Math.round(Math.min(p[0],drag.p0[0])),Math.round(Math.min(p[1],drag.p0[1])),Math.round(Math.max(p[0],drag.p0[0])),Math.round(Math.max(p[1],drag.p0[1]))];
 if(b[2]>b[0]&&b[3]>b[1]){let n=gt.items.length+1;const ids=new Set(gt.items.map(i=>i.id));while(ids.has('h'+n))n++;
 const cy=(b[1]+b[3])/2,ln=gt.lines.find(l=>cy>=l.box[1]&&cy<=l.box[3]);gt.items.push({id:'h'+n,box:b,line:ln?ln.id:null,kind:'glyph',label:null,verified:true,uncertain:false,same_as:null,proposed:null});sel=gt.items.length-1;}
 mode=null;}drag=null;panel();draw();};
function upd(){const it=gt.items[sel];if(!it)return;it.label=$('label').value||null;it.kind=$('kind').value;it.line=$('line').value||null;it.same_as=$('same').value||null;it.verified=$('ver').checked;it.uncertain=$('unc').checked;it.outline=$('outl').checked;draw();}
['label','kind','line','same','ver','unc','outl'].forEach(id=>$(id).oninput=upd);$('zoom').oninput=draw;
$('new').onclick=()=>{mode='new';};$('del').onclick=()=>{if(sel>=0){gt.items.splice(sel,1);sel=-1;panel();draw();}};
$('prev').onclick=()=>{sel=Math.max(0,sel-1);panel();draw();};$('next').onclick=()=>{sel=Math.min(gt.items.length-1,sel+1);panel();draw();};
$('vline').onclick=()=>{const it=gt.items[sel];if(!it||!it.line)return;const ln=gt.lines.find(l=>l.id===it.line);if(ln)ln.verified=true;draw();};
$('save').onclick=async()=>{const r=await fetch('/gt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(gt)});$('msg').textContent=await r.text();};
document.onkeydown=e=>{if(e.target.tagName==='INPUT'&&e.target.type==='text')return;const k=e.key;if(k==='n')$('new').click();else if(k==='Delete'||k==='Backspace')$('del').click();
 else if(k==='s'){e.preventDefault();$('save').click();}else if(k==='ArrowRight')$('next').click();else if(k==='ArrowLeft')$('prev').click();
 else if(k==='v'){$('ver').checked=!$('ver').checked;upd();}else if(k==='u'){$('unc').checked=!$('unc').checked;upd();}};
load();
</script></body></html>"""


def make_handler(image_path: Path, gt_path: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/image":
                self._send(200, image_path.read_bytes(), mimetypes.guess_type(image_path.name)[0] or "image/png")
            elif self.path == "/gt":
                self._send(200, gt_path.read_text(encoding="utf-8"), "application/json; charset=utf-8")
            else:
                self._send(404, "not found")

        def do_POST(self):
            if self.path != "/gt":
                return self._send(404, "not found")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                gt = G.validate(json.loads(self.rfile.read(n).decode("utf-8")))
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, f"保存しなかった: {e}")
            G.save(gt, gt_path)
            st = G.status(gt)
            self._send(200, f"保存した: {gt_path.name}\n記号 {st['glyphs']}（確認済み {st['verified_glyphs']}）、"
                            f"字形 {st['labels']} 種、不確か {st['uncertain']}、文字以外 {st['non_glyph']}")
    return H


def prepare(image, gt_path, report=None):
    gt_path = Path(gt_path)
    if not gt_path.exists():
        if report:
            rep = json.loads(Path(report).read_text(encoding="utf-8"))
            gt = G.draft_from_report(rep, Path(image).name)
        else:
            gt = G.validate({"schema": G.SCHEMA, "image": Path(image).name, "source": "human", "lines": [], "items": []})
        G.save(gt, gt_path)
    return G.load(gt_path)


def serve(image, gt_path, report=None, port=8765, open_browser=False):
    prepare(image, gt_path, report)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Path(image), Path(gt_path)))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"アノテーション UI: {url}  （Ctrl+C で終了。保存先 {gt_path}）")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
