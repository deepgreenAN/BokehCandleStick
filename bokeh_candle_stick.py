import pandas as pd
import numpy as np
import datetime

import math
from pathlib import Path

import bokeh.plotting
import bokeh.io
from bokeh.io import push_notebook

from bokeh.models import ColumnDataSource, BooleanFilter, CDSView, Range1d
from bokeh.models import DatetimeTickFormatter
from bokeh.io import curdoc


from df_transforms_ver1 import get_df_freq, get_sec_from_freq, get_next_datetime, ConvertFreqOHLCV
from df_transforms_ver1 import middle_sample_type_with_check


def static_candlestick(df, ohlc_dict, freq_str=None):
    """
    静的にロウソク足チャートを描画．入力がpandas.DataFrameであることに注意
    df: pandas.DataFrame
        株価データのDataFrame．銘柄は一つに絞らなくてもよい．
    ohlc_dict: dict of str
        {"Open":カラム名,"Close":カラム名}のような辞書，描画したい銘柄の始値，終値，高値，安値を指定する．    
    freq_str: str
        描画のサンプリング周期

    Returns
    -------
    p: bokeh.plotting.figure
        描画されたfigure
    """
    if freq_str is None:
        freq_str = get_df_freq(df)
    
    df = df.copy()  # index等を変更するため
    
    # 同じdatetmeを持つnaiveなdatetimeに変形
    if df.index.tzinfo is not None:  # awareな場合
        df.index = df.index.tz_localize(None)
        
    convert = ConvertFreqOHLCV(freq_str)
    df = convert(df)
        
    seconds = get_sec_from_freq(freq_str)
        
    if set(list(ohlc_dict.keys())) < set(["Open", "High", "Low", "Close"]):
           raise ValueError("keys of ohlc_dict must have 'Open', 'High', 'Low', 'Close'.")
    elif set(list(ohlc_dict.keys())) > set(["Open", "High", "Low", "Close", "Volume"]):  #Volumeは別にあってもよい
        raise ValueError("keys of ohlc_dict is too many.")
    
    increase = df[ohlc_dict["Close"]] >= df[ohlc_dict["Open"]]  # ポジティブになるインデックス
    decrease = df[ohlc_dict["Open"]] > df[ohlc_dict["Close"]]  # ネガティブになるインデックス
    width = seconds*1000  # 分足なので，60秒,1000は？　micro second単位
    
    p = bokeh.plotting.figure(x_axis_type="datetime", plot_width=1000)
    
    p.segment(df.index, df[ohlc_dict["High"]], df.index, df[ohlc_dict["Low"]],
              color="black"
             )
    p.vbar(df.index[increase],
           width,
           df[ohlc_dict["Open"]][increase],
           df[ohlc_dict["Close"]][increase],
           fill_color="#4be639", line_color="black"
          )  # positive
    p.vbar(df.index[decrease],
           width,
           df[ohlc_dict["Close"]][decrease],
           df[ohlc_dict["Open"]][decrease],
           fill_color="#F2583E", line_color="black"
          )  # negative
    
    return p

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
        start_datetime: datetime.datetime
            開始時刻
        end_datetime: datetime.datetime
            終了時刻
        Returns
        -------
        start_df: pandas.DataFrame
            描画初期のデータ．データフレームとする
        """
        start_df_raw = self.stock_df[(self.stock_df.index >= start_datetime) & (self.stock_df.index < end_datetime)].copy()  # 一応コピー
        start_df = self.converter(start_df_raw)
        return start_df
    
    def iter_data(self, start_datetime):
        """
        データを一つ一つ取得するためジェネレータ
        start_datetime: datetime.datetime
            イテレーションの開始時の取得する時刻
        yield
        -------
        one_df: pandas.DataFrame
            取り出されたデータ．リサンプリングされた後，長さ1のデータになる
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

class BokehCandleStick:
    def __init__(self, 
                 stock_data_supplier,  
                 ohlc_dict, 
                 initial_start_datetime, 
                 initial_end_datetime, 
                 freq_str="T", 
                 figure=None,
                 y_axis_margin=50, 
                 use_x_range=True,
                 use_y_range=True,
                 data_left_times=1,
                 is_notebook=True,
                 use_formatter=True
                ):
        """
        stock_supplier: StockDataSupplier or any
            株価データを供給するためのオブジェクト
        ohlc_dict: dict of str
            {"Open":カラム名,"Close":カラム名}のような辞書，stock_dbの出力に依存する
        initial_start_date: datetime
            開始時のx_rangeの下限のdatetime
        initial_end_date: datetime
            開始じのx_rangeの上限のdatetime
        freq_str: str
            サンプリング周期
        figure: bokeh.plotting.Figure
            複数描画の場合
        y_axis_margin: int
            yの表示領域のマージン
        use_x_range: bool
            このクラスにx_rangeの変更を任せるかどうか        
        """    
        self.stock_data_supplier = stock_data_supplier
        self.ohlc_dict = ohlc_dict
        self.y_axis_margin = y_axis_margin
        self.is_notebook = is_notebook
        self.t = None
        self.use_x_range = use_x_range
        self.use_y_range = use_y_range
        self.use_formatter = use_formatter
        
        # ymax, yminを整えるのに使う
        self.last_ymax = self.y_axis_margin
        self.last_ymin = - self.y_axis_margin
        
        self.freq_str = middle_sample_type_with_check(freq_str)

        seconds = get_sec_from_freq(self.freq_str)

        # ohlc_dictのチェック
        if set(list(ohlc_dict.keys())) < set(["Open", "High", "Low", "Close"]):
            raise ValueError("keys of ohlc_dict must have 'Open', 'High', 'Low', 'Close'.")
        elif set(list(ohlc_dict.keys())) > set(["Open", "High", "Low", "Close", "Volume"]):  #Volumeは別にあってもよい
            raise ValueError("keys of ohlc_dict is too many.")

        # 最初のDataFrame
        start_df = self.stock_data_supplier.initial_data(start_datetime=initial_start_datetime, end_datetime=initial_end_datetime)
                
        # 部分DataFrame(OHLC)を取得
        self.ohlc_column_list = [self.ohlc_dict["Open"], self.ohlc_dict["High"], self.ohlc_dict["Low"], self.ohlc_dict["Close"]]
        sub_start_df = start_df.loc[:,self.ohlc_column_list]
        self.initial_length = len(sub_start_df.index)
        self.source_length = self.initial_length * data_left_times
        
        # bokehの設定
        initial_increase = sub_start_df[self.ohlc_dict["Close"]] >= sub_start_df[self.ohlc_dict["Open"]]  # ポジティブになるインデックス
        initial_decrease = sub_start_df[self.ohlc_dict["Open"]] > sub_start_df[self.ohlc_dict["Close"]]  # ネガティブになるインデックス
        width = seconds*1000  # 分足なので，60秒,1000は？　micro second単位
        
        sub_start_df = self._fill_nan_zero(sub_start_df)
        
        # 同じdatetmeを持つnaiveなdatetimeに変形
        if sub_start_df.index.tzinfo is not None:  # awareな場合
            sub_start_df.index = sub_start_df.index.tz_localize(None)
        #print("sub_start_df:",sub_start_df)
        
        self.source = ColumnDataSource(sub_start_df)
        
        increase_filter = BooleanFilter(initial_increase)
        decrease_filter = BooleanFilter(initial_decrease)

        self.view_increase = CDSView(source=self.source, filters=[increase_filter,])
        self.view_decrease = CDSView(source=self.source, filters=[decrease_filter,])
        
        y_max, y_min = self._make_y_range(sub_start_df, margin=self.y_axis_margin)
        
        if figure is None:  # コンストラクタにbokehのfigureが与えられない場合
            if not self.use_x_range or not self.use_y_range:
                raise ValueError("set the use_x_range: True, use_y_range: True")
            source_df = self.source.to_df()
            timestamp_series = source_df.loc[:,"timestamp"]
            self.x_range = Range1d(timestamp_series.iloc[-self.initial_length], timestamp_series.iloc[-1])  # 最後からinitial_length分だけ表示させるためのx_range
            #print("x_range:",self.x_range.start, self.x_range.end)
            self.y_range = Range1d(y_min, y_max)
            self.dp = bokeh.plotting.figure(x_axis_type="datetime", plot_width=1000, x_range=self.x_range, y_range=self.y_range)
        else:
            self.dp = figure
            self.y_range = figure.y_range
            self.x_range = figure.x_range
        
        self.dp.segment(x0="timestamp", y0=self.ohlc_dict["Low"], x1="timestamp", y1=self.ohlc_dict["High"],
                        source=self.source, line_color="black"
                        )  # indexはインデックスの名前で指定されるらしい

        self.dp.vbar(x="timestamp",
                     width=width,
                     top=self.ohlc_dict["Open"],
                     bottom=self.ohlc_dict["Close"],
                     source=self.source, 
                     view=self.view_increase,
                     fill_color="#4be639",
                     line_color="black")  # positive

        self.dp.vbar(x="timestamp",
                     width=width,
                     top=self.ohlc_dict["Close"],
                     bottom=self.ohlc_dict["Open"],
                     source=self.source, 
                     view=self.view_decrease,
                     fill_color="#F2583E",
                     line_color="black")  # negative
        
        # formatter 機能しない
        if self.use_formatter:
            x_format = "%m-%d-%H-%M"
            self.dp.xaxis.formatter = DatetimeTickFormatter(
                minutes=[x_format],
                hours=[x_format],
                days=[x_format],
                months=[x_format],
                years=[x_format]
            )
            self.dp.xaxis.major_label_orientation = math.radians(45)

            
        self.temp_increase = initial_increase
        self.temp_decrease = initial_decrease
        
        # データ供給用ジェネレータ
        self.stock_data_supplier_gen = self.stock_data_supplier.iter_data(start_datetime=initial_end_datetime)
    
    def update(self):
        
        # ソースに加える長さ1のDataFrame
        one_df = next(self.stock_data_supplier_gen)  # ジェネレーターから取り出す
        one_df = self._fill_nan_zero(one_df)  # Noneをなくしておく(bokehが認識できるようにするため)
                
        # 同じdatetimeの値をもつnaiveなdatetimeを取得：
        if len(one_df.index) > 0:
            one_df.index = one_df.index.tz_localize(None)
            
        new_dict = {i:[one_df.loc[one_df.index[0],i]] for i in self.ohlc_column_list}

        #print("new_dict:",new_dict)
        new_dict["timestamp"] = np.array([one_df.index[0].to_datetime64()])

        # filterの調整
        open_valaue = one_df.loc[one_df.index[0], self.ohlc_dict["Open"]]
        close_value = one_df.loc[one_df.index[0], self.ohlc_dict["Close"]]
        
        if open_valaue is not None and close_value is not None:
            inc_add_bool_df = pd.Series([open_valaue<=close_value],index=one_df.index)  # ポジティブになるインデックス
            dec_add_bool_df = pd.Series([open_valaue>close_value],index=one_df.index)  # ネガティブになるインデックス
        else:
            inc_add_bool_df = pd.Series([False],index=one_df.index)
            dec_add_bool_df = pd.Series([False],index=one_df.index)

        new_increase_booleans = pd.concat([self.temp_increase, inc_add_bool_df])  # 後ろに追加
        if len(new_increase_booleans.index) > self.source_length:  # ソースの長さを超えた場合
            new_increase_booleans = new_increase_booleans.drop(new_increase_booleans.index[0])  # 最初を削除
        self.temp_increase = new_increase_booleans

        new_decrease_booleans = pd.concat([self.temp_decrease, dec_add_bool_df])  # 後ろに追加
        if len(new_decrease_booleans.index) > self.source_length:  # ソースの長さを超えた場合
            new_decrease_booleans = new_decrease_booleans.drop(new_decrease_booleans.index[0])  # 最初を削除
        self.temp_decrease = new_decrease_booleans

        # sourceの変更
        self.source.stream(new_data=new_dict, rollover=self.source_length)
        
        # filterの変更
        self.view_increase.filters = [BooleanFilter(self.temp_increase),]
        self.view_decrease.filters = [BooleanFilter(self.temp_decrease),]
           
        # 範囲選択
        source_df = self.source.to_df()
        # yの範囲
        if self.use_y_range:
            y_max, y_min = self._make_y_range(source_df, self.y_axis_margin)
            self.y_range.start = y_min
            self.y_range.end = y_max
        #print("y_range:", self.y_range.start, self.y_range.end)
        # xの範囲
        if self.use_x_range:
            timestamp_series = source_df.loc[:,"timestamp"]
            self.x_range.start = timestamp_series.iloc[-self.initial_length]
            self.x_range.end = timestamp_series.iloc[-1]
        #print("x_range:",self.dp.x_range.start, self.dp.x_range.end)
        
        if self.is_notebook:
            if self.t is None:  # tがセットされていない場合
                raise ValueError("self.t is not setted.")
            push_notebook(handle=self.t)
        
    def _make_y_range(self, df, margin=50):
        new_df = df.replace(0, None)  # Noneに変更してhigh, lowを計算しやすくこれでも0になることがあるらしい．
     
        y_max = new_df.loc[:,self.ohlc_dict["High"]].max(axis=0) + margin
        y_min = new_df.loc[:,self.ohlc_dict["Low"]].min(axis=0) - margin
        
        if y_max == margin:  # Highが0の場合
            y_max = self.last_ymax
        else:
            self.last_ymax = y_max
        
        if y_min == -margin:  # Lowが0の場合
            y_min = self.last_ymin
        else:
            self.last_ymin = y_min
        
        return y_max, y_min
        
    def _fill_nan_zero(self, df):
        return_df = df.fillna(0)
        return return_df
    
    def set_t(self, t):
        self.t = t


class BokehCandleStickDf(BokehCandleStick):
    def __init__(self, 
                 stock_df,  
                 ohlc_dict, 
                 initial_start_datetime, 
                 initial_end_datetime, 
                 freq_str="T", 
                 figure=None,
                 y_axis_margin=50, 
                 use_x_range=True,
                 use_y_range=True,
                 data_left_times=1,
                 is_notebook=True,
                 use_formatter=True
                ):
        """
        stock_df: pandas.DataFrame
            株価データのDataFrame
        ohlc_dict: dict of str
            {"Open":カラム名,"Close":カラム名}のような辞書，stock_dbの出力に依存する
        initial_start_date: datetime
            開始時のx_rangeの下限のdatetime
        initial_end_date: datetime
            開始じのx_rangeの上限のdatetime
        freq_str: str
            サンプリング周期
        figure: bokeh.plotting.Figure
            複数描画の場合
        y_axis_margin: int
            yの表示領域のマージン
        use_x_range: bool
            このクラスにx_rangeの変更を任せるかどうか        
        """  
        stock_data_supplier = StockDataSupplier(stock_df, freq_str)
        
        super(BokehCandleStickDf, self).__init__(stock_data_supplier=stock_data_supplier,
                                                 ohlc_dict=ohlc_dict,
                                                 initial_start_datetime=initial_start_datetime,
                                                 initial_end_datetime=initial_end_datetime,
                                                 freq_str=freq_str,
                                                 figure=figure,
                                                 y_axis_margin=y_axis_margin,
                                                 use_x_range=use_x_range,
                                                 use_y_range=use_y_range,
                                                 data_left_times=data_left_times,
                                                 is_notebook=is_notebook,
                                                 use_formatter=use_formatter
                                                )


if __name__ == "__main__":
    from yahoo_stock_reader_ver1 import YahooFinanceStockLoaderMin
    
    from tornado.ioloop import IOLoop  # サーバーをたてるのに必要
    from bokeh.server.server import Server  # サーバーを立てるのに必要

    from pytz import timezone

    def modify_doc(doc):
        stock_names = ["4755.T"]  # 楽天
        stockloader = YahooFinanceStockLoaderMin(stock_names, stop_time_span=2.0, is_use_stop=False)
        stock_df = stockloader.load()

        day_before = datetime.date.today() - datetime.timedelta(days=0)  # 今日のデータはないので，一昨日

        # 日時の取得
        jst_timezone = timezone("Asia/Tokyo")
        #start_time = jst_timezone.localize(datetime.datetime(day_before.year, day_before.month, day_before.day, 9, 0, 0))
        start_time = jst_timezone.localize(datetime.datetime(day_before.year, day_before.month, day_before.day, 12, 30, 0))
        #end_time = jst_timezone.localize(datetime.datetime(day_before.year, day_before.month, day_before.day, 12, 30, 0))
        end_time = jst_timezone.localize(datetime.datetime(day_before.year, day_before.month, day_before.day, 15, 0, 0))

        ohlc_dict = {"Open":"Open_4755", "High":"High_4755", "Low":"Low_4755", "Close":"Close_4755"}
        
        #sub_stock_df = stock_df[(stock_df.index>=start_time)&(stock_df.index<end_time)].copy()  # locとスライスを利用すると両端を含めてしまうため
        #converter = ConvertFreqOHLCV(freq_str="5T")
        #resampled_df = converter(sub_stock_df.copy())
        #p = static_candlestick(resampled_df, ohlc_dict)
        #bokeh.io.show(p)
        #curdoc().add_root(p)

        bokeh_candle_stick = BokehCandleStickDf(stock_df,  
                                                ohlc_dict=ohlc_dict, 
                                                initial_start_datetime=start_time,
                                                initial_end_datetime=end_time,
                                                freq_str="5T",
                                                y_axis_margin=10,
                                                data_left_times=5,
                                                use_formatter=False,
                                                is_notebook=False
                                            )

        doc.add_root(bokeh_candle_stick.dp)
        doc.add_periodic_callback(bokeh_candle_stick.update, 1000)

    server = Server({'/bkapp': modify_doc}, io_loop=IOLoop(),port=5006)
    server.start()
    server.io_loop.start()

    