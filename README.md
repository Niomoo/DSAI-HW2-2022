# DSAI-HW-2022
## Execution
```
python trader.py --training training.csv --testing testing.csv --output output.csv
```
## Datasets
### Resource
助教所提供的NASDAQ:IBM指數，四項欄位分別 **open-high-low-close**
- [Training data](https://www.dropbox.com/s/uwift61i6ca9g3w/training.csv?dl=0)
    ![](https://i.imgur.com/1ctSNfT.png)

- [Testing data](https://www.dropbox.com/s/duqiffdpcadu6s7/testing.csv?dl=0)
    ![](https://i.imgur.com/sP9N1AF.png)

### Visualization
1. Training Data 整體的價格趨勢
    ![](https://i.imgur.com/laSd8ED.png)
    - 可看出資料整體具有季節性趨勢
    - 價格可能發生不正常的急劇下降
2. Open price 的時間分析
    ![](https://i.imgur.com/0bp82qN.png)
    - 額外拿出開盤價格觀察
    - 可看出資料具有季節性的變化
3.  Close price 的時間分析
    ![](https://i.imgur.com/lUTaRkx.png)
    - 額外拿出收盤價格觀察
    - 可看出資料具有季節性的變化
## Trading
### Trading Strategy
1. 每天進行1筆交易，交易行為分為1(Buy)、0(NoAction)、-1(Sell)
    - 每筆交易為1單位
    - 持股上限為1單位、下限為-1單位
    - 收益使用開盤價計算

2. 根據每天持股狀態與預測的開盤價計算
    - 若預測開盤價 > 前一天開盤價，表示隔天會漲
        - 持股狀態為1，output=0 (NoAction)
        - 持股狀態為0，output=1 (Buy)
        - 持股狀態為-1，output=1 (Buy)
    - 若預測開盤價 = 前一天開盤價，表示隔天平盤
        - 持股狀態為1，output=0 (NoAction)
        - 持股狀態為0，output=0 (NoAction)
        - 持股狀態為-1，output=0 (NoAction)
    - 若預測開盤價 < 前一天開盤價，表示隔天會跌
        - 持股狀態為1，output=-1 (Sell)
        - 持股狀態為0，output=-1 (Sell)
        - 持股狀態為-1，output=0 (NoAction)
    - 最後一天使用收盤價賣出目前持股，持有1單位賣出、持有-1單位買入，使持股歸零

## Training
### Preprocessing
- 將 training data 的開盤價進行正規化
    ![](https://i.imgur.com/1Nt0q5s.png)
### LSTM Model
- 模型結構
    ![](https://i.imgur.com/x3qG2sD.png)
- training loss
    ![](https://i.imgur.com/DVzKYbj.png)
## Result
- testing data 的預測結果
    ![](https://i.imgur.com/4CWFoHA.png)
- output
    - 存在 output.csv 中
    - 透過 profit_calculator 計算出的結果可達到 5.770000000000039