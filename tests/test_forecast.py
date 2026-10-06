import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from forecast import sequences, forecast, metrics, SalesLSTM

class IntegrityTests(unittest.TestCase):
    def test_sequence_targets_follow_history(self):
        x,y=sequences(np.arange(35,dtype=float))
        self.assertEqual(tuple(x.shape),(5,30,1))
        np.testing.assert_array_equal(x[0,:,0].numpy(),np.arange(30))
        self.assertEqual(y[0].item(),30)
        self.assertEqual(y[-1].item(),34)

    def test_recursive_forecast_uses_own_predictions(self):
        class Next(torch.nn.Module):
            def forward(self,x): return x[:,-1,:]+0.1
        scaler=MinMaxScaler().fit(np.array([0,10]).reshape(-1,1))
        result=forecast(Next(),scaler,np.zeros(30),3,torch.device('cpu'))
        np.testing.assert_allclose(result,[1,2,3],rtol=1e-6)

    def test_metrics_original_units(self):
        result=metrics([10,20],[12,16])
        self.assertEqual(result['MAE'],3)
        self.assertAlmostEqual(result['RMSE'],np.sqrt(10))

    def test_insufficient_history_rejected(self):
        with self.assertRaises(ValueError): sequences(np.arange(30))

    def test_lstm_output_shape(self):
        self.assertEqual(tuple(SalesLSTM()(torch.zeros(4,30,1)).shape),(4,1))

if __name__=='__main__': unittest.main()
