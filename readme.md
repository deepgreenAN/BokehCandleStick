# Bokehによる動的なロウソク足チャート

リアルタイム更新を念頭に，Bokehを利用して動的に変化するロウソク足チャートを描画する．
jupyter notebookとbokeh serverの両方で利用できる．

以下，bokehサーバーでのデモ
```
python bokeh_candle_stick.py
```
あるいは
```
bokeh serve bokeh_candle_stick_main.py
```
jupyter のデモと同様に楽天の一昨日からの5分足のロウソク足チャートを描画している．
コードの詳細はbokeh_candlestick.ipynbに記載

<img src="https://www.dropbox.com/s/oh8rjwc21r1i6md/candle_stick.gif?raw=1" alt="ロウソク足チャート" title="ロウソク足チャート">

## jupyter notebookでの使い方
使い方例
```python
from pathlib import Path
from pytz import timezone
import datetime
import time

from df_transforms_ver1 import get_next_datetime, ConvertFreqOHLCV
from yahoo_stock_reader_ver1 import YahooFinanceStockLoaderMin
from bokeh_candle_stick import BokehCandleStick

from bokeh.io import output_notebook
import bokeh.io
output_notebook()

# カスタムDataSupplierの定義(これはお好み)
class StockDataSupplier():
    """
    BokehCandleStickクラスに渡す，ロウソク足チャートの描画のためのデータ供給クラス．
    自作する場合，このクラスを継承する必要は無いが，
    initial_data(描画開始時に描画するデータを返す)メソッドと
    iter_data(一つ一つデータを返す)メソッドの二つを実装している必要がある．
    """
    def __init__(self, df, freq_str):
        self.stock_df = df
        self.freq_str = freq_str
        self.converter = ConvertFreqOHLCV(self.freq_str)  # サンプリング周期のコンバーター

    def initial_data(self, start_datetime, end_datetime):
        """
        描画初期のデータを取得するためのメソッド
        """
        start_df_raw = self.stock_df[(self.stock_df.index >= start_datetime) & (self.stock_df.index < end_datetime)].copy()  # 一応コピー
        start_df = self.converter(start_df_raw)
        return start_df
    
    def iter_data(self, start_datetime):
        """
        データを一つ一つ取得するためジェネレータ
        """
        temp_start_datetime = start_datetime  # 1イテレーションにおける開始時間
        while True:
            temp_end_datetime = get_next_datetime(temp_start_datetime, freq_str=self.freq_str)  # 1イテレーションにおける終了時間
            one_df_raw = self.stock_df[(self.stock_df.index >= temp_start_datetime) & (self.stock_df.index < temp_end_datetime)].copy()  # 変更するので，コピー
            if len(one_df_raw.index) < 1:  #empty dataframeの場合
                one_df_raw.loc[temp_start_datetime] = None  #Noneで初期化
                one_df_resampled = one_df_raw  # 長さ1なので，リサンプリングはしない
            else:
                one_df_resampled = self.converter(one_df_raw)  # リサンプリング
            yield one_df_resampled
            temp_start_datetime = temp_end_datetime  # 開始時間を修正


# 株価データの取得(これはお好み)
stock_names = ["4755.T"]  # 楽天
stockloader = YahooFinanceStockLoaderMin(stock_names, stop_time_span=2.0, is_use_stop=False)
stock_df = stockloader.load()

# 開始時刻の設定
day_before = datetime.date.today() - datetime.timedelta(days=2)  # 今日のデータはないので，一昨日

jst_timezone = timezone("Asia/Tokyo")
start_time = jst_timezone.localize(datetime.datetime(day_before.year, day_before.month, day_before.day, 12, 30, 0))
end_time = jst_timezone.localize(datetime.datetime(day_before.year, day_before.month, day_before.day, 15, 0, 0))

# カスタムDataSupplierの作成
stock_data_supplier = StockDataSupplier(stock_df, freq_str="5T")

# 動的ロウソク足チャートの描画
ohlc_dict = {"Open":"Open_4755", "High":"High_4755", "Low":"Low_4755", "Close":"Close_4755"}
bokeh_candle_stick = BokehCandleStick(stock_data_supplier,  
                                      ohlc_dict, 
                                      initial_start_datetime=start_time,
                                      initial_end_datetime=end_time,
                                      freq_str="5T",
                                      y_axis_margin=10,
                                      data_left_times=5,
                                      use_formatter=True,
                                      is_notebook=True
                                      )


t = bokeh.io.show(bokeh_candle_stick.dp, notebook_handle=True)
bokeh_candle_stick.set_t(t)
for i in range(1000):
    time.sleep(1)
    bokeh_candle_stick.update()                
```
