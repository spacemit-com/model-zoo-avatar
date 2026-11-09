"""Model-free tests of actual process cancellation, timeout and instance ownership."""
import importlib.util,os,sys,tempfile,threading,time,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from spacemit_avatar.pipeline import AvatarService

@unittest.skipUnless(os.name=='posix','Native process lifecycle targets Linux')
class ProcessLifecycle(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.worker=self.root/'worker with spaces';self.pidfile=self.root/'pid'
  self.worker.write_text('#!/usr/bin/env python3\nimport os,time\nfrom pathlib import Path\nPath('+repr(str(self.pidfile))+').write_text(str(os.getpid()))\ntime.sleep(30)\n');self.worker.chmod(0o755)
  self.config=self.root/'config.yaml';self.config.write_text('schema: 1\nhubert:\n  model: unused\n  timeout_s: 5\nunet:\n  model: unused\n')
  self.audio=self.root/'sample.wav';self.audio.write_bytes(b'placeholder: worker does not inspect model/audio')
 def tearDown(self):self.tmp.cleanup()
 def wait_pid(self):
  for _ in range(200):
   if self.pidfile.exists():
    try:return int(self.pidfile.read_text())
    except ValueError:pass
   time.sleep(.01)
  self.fail('Worker never started')
 def assert_dead(self,pid):
  with self.assertRaises(ProcessLookupError):os.kill(pid,0)
 def test_cancel_holds_instance_until_child_reaped(self):
  engine=AvatarService(self.config,self.worker);errors=[]
  def run():
   try:engine.render_file(self.audio,self.root/'frames')
   except Exception as e:errors.append(e)
  thread=threading.Thread(target=run);thread.start();pid=self.wait_pid()
  with self.assertRaises(RuntimeError):engine.render_file(self.audio,self.root/'other')
  engine.cancel();thread.join(3);self.assertFalse(thread.is_alive());self.assertTrue(errors);self.assert_dead(pid);engine.close()
 def test_timeout_kills_and_reaps_child(self):
  self.config.write_text('schema: 1\nhubert:\n  model: unused\n  timeout_s: 0.5\nunet:\n  model: unused\n')
  with AvatarService(self.config,self.worker) as engine:
   with self.assertRaises(TimeoutError):engine.render_file(self.audio,self.root/'frames')
  self.assert_dead(int(self.pidfile.read_text()))
 def test_closed_and_existing_output_are_rejected(self):
  engine=AvatarService(self.config,self.worker);out=self.root/'frames';out.mkdir();(out/'keep.png').write_bytes(b'keep')
  with self.assertRaises(FileExistsError):engine.render_file(self.audio,out)
  self.assertEqual((out/'keep.png').read_bytes(),b'keep');engine.close()
  with self.assertRaises(RuntimeError):engine.render_file(self.audio,self.root/'new')
 def test_nonfinite_timeout_rejected(self):
  self.config.write_text('schema: 1\nhubert:\n  model: unused\n  timeout_s: .nan\nunet:\n  model: unused\n')
  with self.assertRaises(ValueError):AvatarService(self.config,self.worker)

if __name__=='__main__':unittest.main()
