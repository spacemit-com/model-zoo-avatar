import contextlib,hashlib,http.server,importlib.util,io,json,os,tarfile,tempfile,threading,unittest,wave
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('downloads',ROOT/'scripts/download_resources.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def item(data):return {'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}

class ResourceContracts(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def archive(self,records):
  path=self.root/'archive.tar.gz'
  with tarfile.open(path,'w:gz') as t:
   for name,data,kind in records:
    e=tarfile.TarInfo(name)
    if kind=='file':e.size=len(data);t.addfile(e,io.BytesIO(data))
    else:e.type=tarfile.SYMTYPE;e.linkname='../outside';t.addfile(e)
  return path
 def test_verified_extract_and_offline_reuse(self):
  data=b'only requested model';e={'member':'models/a.onnx','dest':'a.onnx',**item(data)}
  a=self.archive([('models/a.onnx',data,'file'),('../../escape',b'unselected','file')]);out=self.root/'out';out.mkdir();d.install(a,'tar.gz',[e],out)
  self.assertEqual((out/'a.onnx').read_bytes(),data);self.assertFalse((self.root/'escape').exists());d.install(self.root/'missing','tar.gz',[e],out)
 def test_existing_modified_file_retained(self):
  out=self.root/'out';out.mkdir();(out/'a').write_bytes(b'private');e={'member':'a','dest':'a',**item(b'expected')}
  with self.assertRaises(FileExistsError):d.install(self.root/'missing','tar.gz',[e],out)
  self.assertEqual((out/'a').read_bytes(),b'private')
 def test_rejects_link_and_traversal(self):
  a=self.archive([('a',b'','link')]);out=self.root/'out';out.mkdir()
  with self.assertRaises(ValueError):d.install(a,'tar.gz',[{'member':'a','dest':'a',**item(b'')}],out)
  with self.assertRaises(ValueError):d.target(out,'../escape')
 def test_rejects_duplicate_and_wrong_hash(self):
  data=b'data';e={'member':'a','dest':'a',**item(data)};out=self.root/'out';out.mkdir();a=self.archive([('a',data,'file'),('a',data,'file')])
  with self.assertRaises(ValueError):d.install(a,'tar.gz',[e],out)
  a=self.archive([('a',b'fail','file')]);other=self.root/'other';other.mkdir()
  with self.assertRaises(ValueError):d.install(a,'tar.gz',[e],other)
  self.assertFalse((other/'a').exists())
 def test_cache_corruption_rejected(self):
  cache=self.root/'cache';cache.mkdir();e={'url':'https://invalid.example/a',**item(b'valid')};p=cache/e['sha256'];p.write_bytes(b'bad')
  with self.assertRaises(ValueError):d.download(e,cache,True)
  self.assertEqual(p.read_bytes(),b'bad')
 def test_http_resume_and_range_ignored(self):
  data=b'0123456789'*1000;requests=[]
  class Handler(http.server.BaseHTTPRequestHandler):
   ignore=False
   def log_message(self,*_):pass
   def do_GET(self):
    header=self.headers.get('Range');requests.append(header);start=int(header[6:-1]) if header and not self.ignore else 0
    self.send_response(206 if start else 200)
    if start:self.send_header('Content-Range',f'bytes {start}-{len(data)-1}/{len(data)}')
    self.send_header('Content-Length',str(len(data)-start));self.end_headers();self.wfile.write(data[start:])
  server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   for ignore in [False,True]:
    Handler.ignore=ignore;cache=self.root/str(ignore);cache.mkdir();e={'url':f'http://127.0.0.1:{server.server_port}/model',**item(data)};(cache/(e['sha256']+'.part')).write_bytes(data[:91]);dest=d.download(e,cache)
    self.assertEqual(dest.read_bytes(),data);self.assertEqual(requests[-1],'bytes=91-')
  finally:server.shutdown();server.server_close();thread.join()
 def test_silence_generated_without_network(self):
  buf=io.BytesIO()
  with wave.open(buf,'wb') as w:w.setparams((1,2,16000,0,'NONE',''));w.writeframes(bytes(32000))
  manifest={'schema':1,'resource_version':'test','archives':{},'assets':{'feifan':[]},'audio':{'silence.wav':{'generated':'silence_16k_1s',**item(buf.getvalue())}}};p=self.root/'manifest.json';p.write_text(json.dumps(manifest));out=self.root/'out'
  d.main(['assets','--manifest',str(p),'--assets-dir',str(out),'--offline']);self.assertEqual((out/'audio/silence.wav').read_bytes(),buf.getvalue())

if __name__=='__main__':unittest.main()
