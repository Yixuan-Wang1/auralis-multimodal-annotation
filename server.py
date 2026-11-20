"""Auralis dataset-backed model API using only Python's standard library."""
from __future__ import annotations
import csv, hashlib, json, mimetypes, os, struct, time, wave
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
DATASET = Path(os.environ.get("AURALIS_DATASET_ROOT", r"G:\avcs"))
RUNTIME = ROOT / "runtime" / "annotations"
API_VERSION = "1.0.0"

class DatasetIndex:
    def __init__(self, root: Path):
        self.root, self.samples = root, {}
        metadata = root / "meta_data.csv"
        if metadata.exists():
            with metadata.open("r", encoding="utf-8-sig", newline="") as f:
                self.samples = {r["video_id"]: r.get("split", "train") for r in csv.DictReader(f) if r.get("video_id")}
    def split_for(self, video_id):
        if video_id not in self.samples: raise KeyError(f"unknown video_id: {video_id}")
        return self.samples[video_id]
    def frames(self, video_id):
        p = self.root / "visual_frames" / video_id
        return sorted(p.glob("*.png"), key=lambda x: int(x.stem)) if p.exists() else []
    def captions(self, video_id):
        p = self.root / "caption" / self.split_for(video_id) / f"{video_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    def audio(self, video_id):
        return self.root / "audio_wav" / self.split_for(video_id) / f"{video_id}.wav"

INDEX = DatasetIndex(DATASET)

def png_size(path):
    if not path: return 640, 360
    h = path.read_bytes()[:24]
    return struct.unpack(">II", h[16:24]) if h[:8] == b"\x89PNG\r\n\x1a\n" else (640, 360)

def nums(key, n=8):
    d = hashlib.sha256(key.encode()).digest()
    return [d[i] / 255 for i in range(n)]

def caption_at(video_id, frame):
    row = next((r for r in INDEX.captions(video_id) if int(r.get("frame", -1)) == frame), {"frame": frame, "caption": ""})
    return row.get("caption", ""), row

def label_for(text):
    t = text.lower()
    for words, label in [(("man","woman","person","people","boy","girl"),"person"),(("dog","cat","horse"),"animal"),(("car","bus","truck"),"vehicle"),(("ball","volleyball","basketball"),"sports_ball"),(("guitar","piano","instrument"),"instrument")]:
        if any(w in t for w in words): return label
    return "foreground_object"

def detection(video_id, frame):
    caption, source = caption_at(video_id, frame)
    files = INDEX.frames(video_id); selected = next((p for p in files if p.stem == str(frame)), files[0] if files else None)
    w, h = png_size(selected); n = nums(f"{video_id}:{frame}:{caption}")
    bw, bh = max(40,int(w*(.22+n[0]*.18))), max(40,int(h*(.35+n[1]*.25)))
    x, y = int(n[2]*max(1,w-bw)), int(n[3]*max(1,h-bh))
    return {"model":"YOLOv11-adapter","mode":"dataset_replay","video_id":video_id,"frame":frame,"image":{"width":w,"height":h,"url":f"/api/v1/media/frame/{video_id}/{frame}"},"detections":[{"id":"det-001","label":label_for(caption),"score":round(.88+n[4]*.1,4),"bbox_xyxy":[x,y,x+bw,y+bh],"source_caption":caption}],"provenance":{"annotation":source,"derived_from":"caption_annotation"}}

def segmentation(video_id, frame, prompt=None):
    d=detection(video_id,frame); x1,y1,x2,y2=d["detections"][0]["bbox_xyxy"]
    poly=[[x1+8,y1],[x2-8,y1],[x2,y1+8],[x2,y2-8],[x2-8,y2],[x1+8,y2],[x1,y2-8],[x1,y1+8]]
    return {"model":"SAM2-adapter","mode":"dataset_replay","video_id":video_id,"frame":frame,"prompt":prompt or {"type":"box","value":[x1,y1,x2,y2]},"masks":[{"id":"mask-001","label":d["detections"][0]["label"],"score":.94,"format":"polygon","polygon":poly}],"provenance":d["provenance"]}

def tracking(video_id,start,end):
    items=[]
    for f in range(start,end+1):
        d=detection(video_id,f)["detections"][0]; items.append({"frame":f,"objects":[{"track_id":"track-001",**d}]})
    return {"model":"Track-Anything-adapter","mode":"dataset_replay","video_id":video_id,"range":[start,end],"tracks":items,"provenance":{"derived_from":"caption_annotation"}}

def alignment(video_id,start=None,end=None):
    rows=INDEX.captions(video_id)
    if start is not None: rows=[r for r in rows if int(r.get("frame",-1))>=int(start)]
    if end is not None: rows=[r for r in rows if int(r.get("frame",-1))<=int(end)]
    events=[]; last=None
    for r in rows:
        text=r.get("caption","")
        if text and text!=last:
            f=int(r["frame"]); events.append({"event_id":f"evt-{len(events)+1:03d}","frame":f,"start_sec":round(f/6,3),"text":text,"confidence":1.0}); last=text
    a=INDEX.audio(video_id); meta={"exists":a.exists(),"url":f"/api/v1/media/audio/{video_id}"}
    if a.exists():
        with wave.open(str(a),"rb") as w: meta.update({"sample_rate":w.getframerate(),"channels":w.getnchannels(),"duration_sec":round(w.getnframes()/w.getframerate(),3)})
    return {"model":"AV-HuBERT+WhisperX-adapter","mode":"dataset_replay","video_id":video_id,"fps":6,"audio":meta,"events":events,"provenance":{"derived_from":"frame_caption_annotation"}}

def sample(video_id):
    frames=INDEX.frames(video_id); caps=INDEX.captions(video_id); nonempty=[r for r in caps if r.get("caption")]
    return {"video_id":video_id,"split":INDEX.split_for(video_id),"frame_count":len(frames),"caption_count":len(nonempty),"audio_available":INDEX.audio(video_id).exists(),"first_frame_url":f"/api/v1/media/frame/{video_id}/{frames[0].stem}" if frames else None,"annotation_preview":nonempty[:3]}

def openapi():
    paths={p:{"post":{"summary":s,"responses":{"200":{"description":"Dataset-backed result"}}}} for p,s in {"/api/v1/models/detection/infer":"Object detection","/api/v1/models/segmentation/infer":"Promptable segmentation","/api/v1/models/tracking/infer":"Cross-frame tracking","/api/v1/models/alignment/infer":"Audio-video alignment","/api/v1/models/caption/infer":"Frame caption","/api/v1/pipeline/run":"Full pipeline","/api/v1/annotations/save":"Save revision"}.items()}
    return {"openapi":"3.0.3","info":{"title":"Auralis Model API","version":API_VERSION},"servers":[{"url":"http://127.0.0.1:8765"}],"paths":paths}

class Handler(SimpleHTTPRequestHandler):
    server_version="AuralisAPI/1.0"
    def __init__(self,*a,**kw): super().__init__(*a,directory=str(DIST),**kw)
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS"); self.send_header("Access-Control-Allow-Headers","Content-Type"); self.send_header("X-Auralis-API-Version",API_VERSION); super().end_headers()
    def do_OPTIONS(self): self.send_response(204); self.end_headers()
    def send_json(self,payload,status=200):
        body=json.dumps(payload,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def fail(self,e,status=400): self.send_json({"error":type(e).__name__,"message":str(e),"api_version":API_VERSION},status)
    def body(self): return json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))) or b"{}")
    def file(self,p):
        if not p.is_file(): return self.fail(FileNotFoundError(p.name),404)
        self.send_response(200); self.send_header("Content-Type",mimetypes.guess_type(p.name)[0] or "application/octet-stream"); self.send_header("Content-Length",str(p.stat().st_size)); self.end_headers()
        with p.open("rb") as f:
            while b:=f.read(262144): self.wfile.write(b)
    def do_GET(self):
        u=urlparse(self.path); p=unquote(u.path); q=parse_qs(u.query)
        try:
            if p=="/api/v1/health": return self.send_json({"status":"ok","api_version":API_VERSION,"dataset_root":str(DATASET),"dataset_available":DATASET.exists(),"samples":len(INDEX.samples),"mode":"dataset_replay"})
            if p=="/api/v1/dataset/stats":
                vals=list(INDEX.samples.values()); return self.send_json({"root":str(DATASET),"samples":len(vals),"splits":{s:vals.count(s) for s in sorted(set(vals))},"modalities":["visual_frames","audio_wav","frame_caption"]})
            if p=="/api/v1/samples":
                split=q.get("split",[None])[0]; limit=min(100,max(1,int(q.get("limit",[12])[0]))); require_frames=q.get("has_frames",["false"])[0].lower()=="true"; ids=[k for k,v in INDEX.samples.items() if split in (None,v) and (not require_frames or bool(INDEX.frames(k)))][:limit]; return self.send_json({"items":[sample(x) for x in ids],"count":len(ids)})
            if p.startswith("/api/v1/samples/"): return self.send_json(sample(p.rsplit("/",1)[-1]))
            if p.startswith("/api/v1/media/frame/"):
                parts=p.split("/"); return self.file(DATASET/"visual_frames"/parts[-2]/f"{int(parts[-1])}.png")
            if p.startswith("/api/v1/media/audio/"): return self.file(INDEX.audio(p.rsplit("/",1)[-1]))
            if p=="/api/v1/openapi.json": return self.send_json(openapi())
            return super().do_GET()
        except KeyError as e: return self.fail(e,404)
        except Exception as e: return self.fail(e)
    def do_POST(self):
        started=time.perf_counter()
        try:
            d=self.body(); vid=d.get("video_id",""); f=int(d.get("frame",0))
            if self.path=="/api/v1/models/detection/infer": r=detection(vid,f)
            elif self.path=="/api/v1/models/segmentation/infer": r=segmentation(vid,f,d.get("prompt"))
            elif self.path=="/api/v1/models/tracking/infer":
                start=int(d.get("start_frame",0)); r=tracking(vid,start,min(int(d.get("end_frame",10)),start+60))
            elif self.path=="/api/v1/models/alignment/infer": r=alignment(vid,d.get("start_frame"),d.get("end_frame"))
            elif self.path=="/api/v1/models/caption/infer":
                text,source=caption_at(vid,f); r={"model":"Caption-adapter","mode":"dataset_replay","video_id":vid,"frame":f,"caption":text,"provenance":{"annotation":source}}
            elif self.path=="/api/v1/pipeline/run": r={"video_id":vid,"mode":"dataset_replay","outputs":{"detection":detection(vid,f),"segmentation":segmentation(vid,f),"tracking":tracking(vid,f,f+5),"alignment":alignment(vid,f,f+30)}}
            elif self.path=="/api/v1/annotations/save":
                INDEX.split_for(vid); RUNTIME.mkdir(parents=True,exist_ok=True); target=RUNTIME/f"{vid}.json"; d["saved_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); target.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8"); r={"saved":True,"video_id":vid,"path":str(target),"revision":hashlib.sha1(target.read_bytes()).hexdigest()[:12]}
            else: return self.fail(FileNotFoundError(self.path),404)
            r["latency_ms"]=round((time.perf_counter()-started)*1000,2); return self.send_json(r)
        except KeyError as e: return self.fail(e,404)
        except Exception as e: return self.fail(e)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--port",type=int,default=8765); ap.add_argument("--dataset-root",default=str(DATASET)); a=ap.parse_args()
    DATASET=Path(a.dataset_root); INDEX=DatasetIndex(DATASET)
    print(f"Auralis API http://{a.host}:{a.port} | dataset={DATASET} | samples={len(INDEX.samples)}",flush=True)
    ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()

