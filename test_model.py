import json,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
from model import initialize,save,forward

class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.folder=Path(cls.tmp.name);cls.w=initialize()
        cls.fp=cls.folder/'fp.bin';cls.q=cls.folder/'q.bin';save(cls.fp,cls.w);save(cls.q,cls.w,True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def run_model(self,path,*args):return json.loads(subprocess.check_output(['./build/pocket','--model',str(path),'--prompt','the ',*args]))
    def test_cpp_matches_numpy_reference(self):
        expected=forward(self.w,np.array([[116,104,101,32]]))[0,-1]
        np.testing.assert_allclose(self.run_model(self.fp,'--logits'),expected,atol=2e-6,rtol=1e-5)
    def test_quantized_logits_close_and_smaller(self):
        fp=np.array(self.run_model(self.fp,'--logits'));q=np.array(self.run_model(self.q,'--logits'))
        self.assertLess(np.max(np.abs(fp-q)),.04)
        self.assertLess(self.q.stat().st_size,self.fp.stat().st_size*.35)
    def test_cached_equals_full_recompute(self):
        for model in (self.fp,self.q):
            self.assertEqual(self.run_model(model,'--steps','30')['text'],self.run_model(model,'--steps','30','--no-cache')['text'])
    def test_reject_bad_file_and_context_overflow(self):
        bad=self.folder/'bad.bin';bad.write_bytes(b'bad')
        for model,steps in ((bad,'1'),(self.fp,'200')):
            p=subprocess.run(['./build/pocket','--model',str(model),'--steps',steps],capture_output=True)
            self.assertNotEqual(p.returncode,0)
    def test_backward_finite_difference(self):
        tokens=np.array([[116,104,101,32]]);targets=np.array([[104,101,32,99]])
        loss,grads=forward(self.w,tokens,targets);eps=.001
        for name,index in [('q',(2,3)),('embedding',(116,3)),('ff1',(2,3)),('position',(1,2)),('head',(99,4))]:
            original=self.w[name][index]
            self.w[name][index]=original+eps;plus=forward(self.w,tokens,targets)[0]
            self.w[name][index]=original-eps;minus=forward(self.w,tokens,targets)[0]
            self.w[name][index]=original
            self.assertAlmostEqual(float(grads[name][index]),(plus-minus)/(2*eps),delta=.001)
if __name__=='__main__':unittest.main()
