import pandas as pd
import numpy as np
import datetime
from pytz import timezone

import math
from pathlib import Path

import bokeh.plotting
import bokeh.io
from bokeh.io import push_notebook

from bokeh.models import ColumnDataSource, BooleanFilter, CDSView, Range1d
from bokeh.models import DatetimeTickFormatter
from bokeh.io import curdoc


from df_transforms_ver1 import get_df_freq, get_sec_from_freq, get_next_datetime, ConvertFreqOHLCV


def static_candlestick(df, ohlc_dict, freq_str=None):
    if freq_str is None:
        freq_str = get_df_freq(df)
    
    df = df.copy()  # index等を変更するため
    
    # 同じdatetmeを持つnaiveなdatetimeに変形
    if df.index.tzinfo is not None:  # awareな場合
        df.index = df.index.tz_localize(None)
        
    convert = ConvertFreqOHLCV(freq_str)
    df = convert(df)  # リサンプリング
        
    seconds = get_sec_from_freq(freq_str)
        
    if set(list(ohlc_dict.keys())) < set(["Open", "High", "Low", "Close"]):
           raise ValueError("keys of ohlc_dict must have 'Open', 'High', 'Low', 'Close'.")
    
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


class BokehCandleStickDf:
    def __init__(self, 
                 stock_df,  
                 ohlc_dict, 
                 initial_start_date, 
                 initial_end_date, 
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
            株価用のデータ．
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
        self.stock_df = stock_df
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
        
        if freq_str is None:
            freq_str = get_df_freq(self.stock_df)
        
        self.freq_str = freq_str

        seconds = get_sec_from_freq(self.freq_str)

        if set(list(ohlc_dict.keys())) < set(["Open", "High", "Low", "Close"]):
               raise ValueError("keys of ohlc_dict must have 'Open', 'High', 'Low', 'Close'.")

        # 最初のDataFrame
        start_df_raw = self.stock_df[(self.stock_df.index >= initial_start_date) & (self.stock_df.index < initial_end_date)]
        self.converter = ConvertFreqOHLCV(self.freq_str)
        start_df = self.converter(start_df_raw.copy())
        
        # 更新の時必要なスタートdatetime
        self.temp_start_datetime = initial_end_date
        
        # 部分DataFrameを取得
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
    
    def update(self):
        # ソースに加える長さ1のDataFrame
        temp_next_datetime = get_next_datetime(self.temp_start_datetime, freq_str=self.freq_str)  # freq_strに従って次の時刻を取得 
        
        one_df_raw = self.stock_df[(self.stock_df.index >= self.temp_start_datetime)&(self.stock_df.index < temp_next_datetime)]
        one_df_resmpled = self.converter(one_df_raw)  # リサンプリング
        
        one_df = self._fill_nan_zero(one_df_resmpled)  # Noneをなくしておく(bokehが認識できるようにするため)
        
        
        # 次の終了時刻を修正
        self.temp_start_datetime = temp_next_datetime
        
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


if __name__ == "__main__":
    from yahoo_stock_reader_ver1 import YahooFinanceStockLoaderMin
    from tornado.ioloop import IOLoop  # サーバーをたてるのに必要
    from bokeh.server.server import Server  # サーバーを立てるのに必要

    def modify_doc(doc):
        stock_names = ["4755.T"]  # 楽天
        stockloader = YahooFinanceStockLoaderMin(stock_names, stop_time_span=2.0, is_use_stop=False)
        stock_df = stockloader.load()

        day_before = datetime.date.today() - datetime.timedelta(days=2)  # 今日のデータはないので，一昨日

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
                                                initial_start_date=start_time,
                                                initial_end_date=end_time,
                                                freq_str="5T",
                                                y_axis_margin=10,
                                                data_left_times=5,
                                                use_formatter=False,
                                                is_notebook=False
                                            )

        print(start_time)
        print("bokeh candle stick",type(bokeh_candle_stick.dp))

        doc.add_root(bokeh_candle_stick.dp)
        doc.add_periodic_callback(bokeh_candle_stick.update, 1000)

    server = Server({'/bkapp': modify_doc}, io_loop=IOLoop(),port=5006)
    server.start()
    server.io_loop.start()

    