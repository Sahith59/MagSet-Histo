"""Build an identifier-free interactive replay of the already verified raw-image demo."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from magset.collect import load_trusted_context, verify_run
from magset.data import file_hash
from scripts.report_breakhis_milestone import verify_demo

HTML = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BreakHis prediction replay</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#12283c;font:17px/1.5 Arial,sans-serif}
main{max-width:1080px;margin:30px auto;padding:30px;background:white;border-radius:14px}
h1{font-size:32px;margin:0 0 8px}h2{font-size:22px;margin:0 0 12px}p{margin:8px 0 18px}
.muted{color:#536578}.badge{display:inline-block;font-size:13px;background:#dcecf0;padding:4px 10px;border-radius:5px}
.grid{display:grid;grid-template-columns:1.5fr 1fr;gap:26px}.panel{border:1px solid #d4dde5;border-radius:10px;padding:22px}
fieldset{border:0;padding:0;margin:22px 0}legend{font-weight:bold;margin-bottom:10px}
label{display:inline-flex;align-items:center;margin:4px 14px 4px 0;padding:8px 10px;background:#eef3f5;border-radius:5px}
input{width:19px;height:19px;margin-right:8px;accent-color:#087e8b}table{border-collapse:collapse;width:100%}
td{padding:6px 0}td:first-child{width:45%}td:last-child{text-align:right;width:18%}
progress{width:100%;height:18px;accent-color:#087e8b}strong{font-size:22px}.score{font-size:34px;color:#087e8b}
.meta{border-top:1px solid #d4dde5;padding-top:16px;margin-top:25px;font-size:13px;overflow-wrap:anywhere}
.truth{font-size:15px;margin-bottom:12px}.empty{padding:20px;background:#fff4dc}button{font:inherit;padding:5px 12px}
[hidden]{display:none!important}@media(max-width:760px){main{margin:0;padding:20px}.grid{grid-template-columns:1fr}h1{font-size:27px}}
</style><main><span class="badge">Recorded validation example • saved model outputs</span>
<h1>Breast histology prediction replay</h1>
<p class="muted">Choose the microscope views available to the model. This viewer replays real, verified outputs; it does not run a model or accept new image uploads.</p>
<fieldset><legend>Available magnifications</legend>
<label><input type="checkbox" value="40" checked>40×</label><label><input type="checkbox" value="100" checked>100×</label>
<label><input type="checkbox" value="200" checked>200×</label><label><input type="checkbox" value="400" checked>400×</label></fieldset>
<div id="empty" class="empty" hidden>Choose at least one magnification.</div>
<div id="results" class="grid"><section class="panel"><h2>Eight subtype outputs</h2>
<p class="truth" id="truth"></p><table aria-label="Subtype probabilities"><tbody id="rows"></tbody></table>
<p>Predicted subtype: <strong id="subtype"></strong></p></section>
<section class="panel"><h2>Benign / malignant secondary output</h2>
<p id="broad" class="score"></p><p id="benign"></p><p id="malignant"></p>
<p class="muted">Malignant score = sum of the four malignant subtype probabilities. Fixed threshold:0.5. This is a different question from naming the exact subtype.</p>
<p id="fields"></p></section></div>
<section class="meta"><p><b>What this demonstrates</b><br>The existing saved Phikon-v2 model processed121 original fields. All15 view combinations reproduced archived validation outputs within5.52e-08. The case was selected lexicographically before inspecting its result.</p>
<p>This is one execution example, not an accuracy estimate or a clinical validation. The original-image computation happened on the cluster; the controls above select its saved outputs.</p>
<p id="source"></p></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),inputs=[...document.querySelectorAll('input')];
const nice=s=>s.replaceAll('_',' '),pct=x=>(100*x).toFixed(1)+'%';
function render(){const mags=inputs.filter(x=>x.checked).map(x=>Number(x.value)).sort((a,b)=>a-b),key=mags.join('+');
document.getElementById('empty').hidden=mags.length>0;document.getElementById('results').hidden=mags.length===0;if(!mags.length)return;
const r=d.predictions[key],p=r.probabilities,pm=[1,3,4,5].reduce((a,i)=>a+p[i],0),pb=[0,2,6,7].reduce((a,i)=>a+p[i],0);
document.getElementById('truth').textContent='Dataset subtype for this example: '+nice(d.truth);
document.getElementById('rows').replaceChildren(...d.class_names.map((name,i)=>{const tr=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td'),c=document.createElement('td'),bar=document.createElement('progress');a.textContent=nice(name);bar.max=1;bar.value=p[i];bar.setAttribute('aria-label',nice(name));b.append(bar);c.textContent=pct(p[i]);tr.append(a,b,c);return tr;}));
document.getElementById('subtype').textContent=nice(r.predicted_subtype);
document.getElementById('broad').textContent=pm>=.5?'Malignant':'Benign';
document.getElementById('benign').textContent='Benign model score: '+pct(pb);document.getElementById('malignant').textContent='Malignant model score: '+pct(pm);
document.getElementById('fields').textContent=mags.reduce((a,m)=>a+d.fields_by_view[String(m)],0)+' original fields across '+mags.length+' available view(s).';
}inputs.forEach(x=>x.addEventListener('change',render));render();
document.getElementById('source').textContent='Saved head SHA256: '+d.model_sha256+' · Source demo SHA256: '+d.raw_demo_sha256;
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','protocol','run-dir','head','raw-demo','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();trusted=load_trusted_context(a.manifest,a.protocol)
    report,_=verify_run(a.run_dir/'metrics.json',trusted)
    archive=[json.loads(line) for line in (a.run_dir/'predictions.jsonl').read_text().splitlines()]
    d=json.loads(a.raw_demo.read_text())
    verify_demo(d,report,trusted,archive,a.head,'9448a7e319167b7417670a882552edb372603fd3')
    value={k:d[k] for k in ('class_names','truth','fields_by_view','predictions','model_sha256')}
    value['raw_demo_sha256']=file_hash(a.raw_demo)
    text=json.dumps(value,allow_nan=False)
    if any(x in text for x in ('SOB_','/Users/','/data/users','/home/users')):raise ValueError('Private identifier in public demo')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(HTML.replace('__DATA__',text))
    print('Verified source replay exported without provider identifiers')


if __name__=='__main__':main()
