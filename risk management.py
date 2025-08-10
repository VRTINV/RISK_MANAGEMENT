import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import random as rd
import xgboost
A="BTCUSD"





def get_close(N):
    rates = mt5.copy_rates_from_pos(A, mt5.TIMEFRAME_H1, 1, N)
    ticks_frame = pd.DataFrame(rates)
    #print(ticks_frame)
    PRICE_array=ticks_frame['close'].to_numpy()
    return PRICE_array

def disconnect():
    mt5.shutdown()
def connect():
    # connect to MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        
        

def isNewTrade(previous):
    current=-1
    connect()
    for p in mt5.positions_get(symbol=A):
        current=p.ticket
        #print(current)
    disconnect()
    return not (current==previous)

def getTicket():
    connect()
    for p in mt5.positions_get(symbol=A):
        current=p.ticket
    disconnect()
    return current

def getPosition():
    connect()
    for p in mt5.positions_get(symbol=A):
        current=p.type  
    disconnect()
    if current==mt5.POSITION_TYPE_BUY:
        return 1 
    if current==mt5.POSITION_TYPE_SELL:
        return -1
ticket=-1 



def pred1(X):
    length=len(X)
    q=5
    xtrain=[]
    ytrain=[]
    for i in range(length-q):
        xtrain.append([1]+X[i:i+q])
        ytrain.append(X[i+q])
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain)
    A=np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(xtrain),xtrain)),np.transpose(xtrain)),ytrain)
    pred=np.matmul(A,np.array([1]+X[length-q:]))
    return pred

def pred(X):
    P=24
    #print("synthetisez data OK")
    res=[]
    for i in range(P):
        res.append(pred1(list(X[i:])+res))
    return np.array(res)


def RETURNS(X):
    n=len(X)
    return np.array([np.log(X[j+1]/X[j]) for j in range(n-1)])



def genDATA(U):
    K=15
    P=24
    n=len(U)
    s=10
    R=RETURNS(U)
    S=np.array([R[j:j+s].std() for j in range(n-s-1)])

    INFR_S=pred(S)
    INFR_R=pred(R)
    
    
    data=[]
    
    for k in range(K):
        line=[U[-1]]
        for p in range(P):
            
            Z=rd.normalvariate()
            line.append(line[-1]*np.exp(INFR_R[p]+INFR_S[p]*Z))
    
        data=np.concatenate((data,line))
    return data

while True:
    if isNewTrade(ticket):
        print(f"NEW TRADE DETECTED")
        ticket=getTicket()
        SIGNAL=getPosition()
        H=25
        D=24
        #model
        connect()
        PRICES=get_close(250)
        disconnect()
        
        X=[]
        Y=[]
        
        for h in range(250-H-D):
            X.append(genDATA(PRICES[h:h+H]))
            Y.append(PRICES[h+H:h+H+D])
            
        XRUN=[]
        hrun=250-H-D
        XRUN.append(genDATA(PRICES[hrun:hrun+H]))
        
        model=xgboost.XGBRegressor(missing=np.inf)
        
        model.fit(X,Y)
        
        YRUN=model.predict(XRUN)[0]


        delta = max (abs(PRICES[-1]-max(YRUN)),abs(PRICES[-1]- min(YRUN)))

        SL=PRICES[-1]-SIGNAL*delta
        TP=PRICES[-1]+SIGNAL*delta
        
        
        
        sl=1*SL
        tp=1*TP
        connect()
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": A,
            "position": ticket,
            "sl": sl,
            "tp": tp,
                }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("Failed to modify order:", result.retcode)
        else:
            print("Order modified successfully")

        disconnect()
