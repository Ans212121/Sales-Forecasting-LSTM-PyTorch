"""Daily aggregate sales forecasting; educational PyTorch pipeline."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SalesLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def sequences(values, lookback=30):
    if len(values) <= lookback:
        raise ValueError('Insufficient history for the lookback window')
    x = np.stack([values[i-lookback:i] for i in range(lookback,len(values))])
    y = values[lookback:]
    return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).reshape(-1,1)

def fit(values, epochs, device, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    scaler = MinMaxScaler().fit(np.asarray(values).reshape(-1,1))
    scaled = scaler.transform(np.asarray(values).reshape(-1,1)).ravel()
    x,y = sequences(scaled)
    loader = DataLoader(TensorDataset(x,y), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = SalesLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    losses=[]
    for epoch in range(epochs):
        model.train(); total=0
        for xb,yb in loader:
            xb,yb = xb.to(device),yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb),yb)
            loss.backward(); optimizer.step()
            total += loss.item()*len(xb)
        losses.append(total/len(x))
        if epoch==0 or (epoch+1)%5==0: print(f'Epoch {epoch+1}/{epochs}: {losses[-1]:.6f}',flush=True)
    return model,scaler,losses

def forecast(model,scaler,values,horizon,device):
    history = list(scaler.transform(np.asarray(values[-30:]).reshape(-1,1)).ravel())
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(history[-30:],dtype=torch.float32,device=device).reshape(1,30,1)
            history.append(float(model(x).item()))
    return np.maximum(scaler.inverse_transform(np.asarray(history[-horizon:]).reshape(-1,1)).ravel(),0)

def metrics(actual,pred):
    residual=np.asarray(pred)-np.asarray(actual)
    return {'MAE':float(np.abs(residual).mean()),'RMSE':float(np.sqrt((residual**2).mean()))}

def run(train_path,test_path,output='.',epochs=30):
    out=Path(output); reports=out/'reports'; reports.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=pd.read_csv(train_path); test=pd.read_csv(test_path)
    if not {'date','sales'}.issubset(train): raise ValueError('train.csv requires date and sales')
    if 'date' not in test: raise ValueError('test.csv requires date')
    train['date']=pd.to_datetime(train['date'],errors='raise')
    train['sales']=pd.to_numeric(train['sales'],errors='raise')
    if train.sales.isna().any() or not np.isfinite(train.sales).all() or (train.sales<0).any():
        raise ValueError('Sales must be finite, nonnegative, and nonmissing')
    daily=train.groupby('date').sales.sum().sort_index()
    calendar=pd.date_range(daily.index.min(),daily.index.max(),freq='D')
    missing=int(len(calendar)-len(daily))
    # Causal fill prevents future observations entering missing historical days.
    daily=daily.reindex(calendar).ffill()
    future=pd.DatetimeIndex(pd.to_datetime(test.date).unique()).sort_values()
    expected=pd.date_range(daily.index[-1]+pd.Timedelta(days=1),periods=len(future))
    if len(future)<15 or not future.equals(expected): raise ValueError('test.csv must include at least 15 consecutive dates immediately after history')
    future=future[:15]
    values=daily.to_numpy(dtype=np.float32)
    if len(values)<=45: raise ValueError('More than 45 daily observations are required')
    history,actual=values[:-15],values[-15:]
    print('Validation training',flush=True)
    model,scaler,losses=fit(history,epochs,device)
    pred=forecast(model,scaler,history,15,device)
    weekly=np.resize(history[-7:],15)
    naive=np.repeat(history[-1],15)
    comparison=pd.DataFrame({'date':daily.index[-15:],'actual_sales':actual,
        'lstm_sales':pred,'weekly_naive_sales':weekly,'last_value_sales':naive})
    comparison['absolute_error']=np.abs(actual-pred)
    comparison.to_csv(reports/'validation_actual_vs_predicted.csv',index=False)
    evidence={'status':'MEASURED_REAL_DATA','seed':42,'epochs':epochs,'device':str(device),
        'versions':{'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'train_rows':len(train),'test_rows':len(test),'daily_observations':len(values),
        'missing_calendar_days_forward_filled':missing,'lookback':30,'horizon':15,
        'validation_start':str(daily.index[-15].date()),'validation_end':str(daily.index[-1].date()),
        'metrics':{'LSTM':metrics(actual,pred),'weekly_naive':metrics(actual,weekly),'last_value':metrics(actual,naive)},
        'raw_sha256':{str(Path(p).name):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [train_path,test_path]}}
    (reports/'metrics.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    plt.figure(figsize=(12,4)); plt.plot(daily.index,daily.values); plt.title('Historical total daily sales'); plt.ylabel('Sales'); plt.tight_layout(); plt.savefig(reports/'historical_sales.png'); plt.close()
    plt.figure(figsize=(10,4)); plt.plot(range(1,epochs+1),losses); plt.title('Validation model training loss'); plt.xlabel('Epoch'); plt.ylabel('Scaled MSE'); plt.tight_layout(); plt.savefig(reports/'training_loss.png'); plt.close()
    plt.figure(figsize=(12,4))
    for col,label in [('actual_sales','Actual'),('lstm_sales','LSTM'),('weekly_naive_sales','Weekly baseline')]: plt.plot(comparison.date,comparison[col],marker='o',label=label)
    plt.legend(); plt.title('15-day holdout: actual vs forecast'); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(reports/'validation.png'); plt.close()
    print('Final training on all historical data',flush=True)
    final,final_scaler,final_losses=fit(values,epochs,device)
    future_pred=forecast(final,final_scaler,values,15,device)
    result=pd.DataFrame({'date':future,'predicted_total_sales':future_pred})
    result.to_csv(reports/'next_15_days_sales_forecast.csv',index=False)
    pd.DataFrame({'epoch':range(1,epochs+1),'validation_training_mse':losses,'final_training_mse':final_losses}).to_csv(reports/'loss_history.csv',index=False)
    torch.save({'state_dict':final.cpu().state_dict(),'scaler_min':final_scaler.min_.tolist(),
        'scaler_scale':final_scaler.scale_.tolist(),'lookback':30,'hidden_size':64,'num_layers':2,
        'seed':42,'epochs':epochs},out/'models/sales_lstm.pth')
    plt.figure(figsize=(12,4)); plt.plot(daily.index[-90:],daily.values[-90:],label='Historical'); plt.plot(future,future_pred,marker='o',label='Forecast'); plt.legend(); plt.xticks(rotation=30); plt.title('Next 15 days: aggregate sales'); plt.tight_layout(); plt.savefig(reports/'forecast.png'); plt.close()
    print(json.dumps(evidence['metrics'],indent=2))
    return evidence,comparison,result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',required=True); parser.add_argument('--test',required=True)
    parser.add_argument('--output',default='.'); parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args()
    if args.epochs<1: parser.error('--epochs must be positive')
    run(args.train,args.test,args.output,args.epochs)
