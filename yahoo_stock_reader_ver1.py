import pandas as pd
import numpy as np
import datetime
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import re
import time



stock_str_list = ['Open', 'High', 'Low', 'Close', 'Volume']

class ColumnNameEditorYahoo():
    """
    yahoo finance apiを利用したときのカラム名を調整するためのクラス
    """
    def __init__(self, extra_str=r".T"):
        self.pattern = re.compile(extra_str)  # とりあえずこれでいいか
        self.column_name_dict = {"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"}
        
    def __call__(self, stock_name, column_name):
        new_column = self.column_name_dict[column_name]
        replaced_stock_name = self.pattern.sub("", stock_name)
        
        return new_column + "_" + replaced_stock_name


class YahooFinanceStockLoaderMin():
    api_last_used_time = None
    api_now_used_time = None
    """
    yahoo finance apiを用いた株価取得ローダー．
    現在から6日まで過去の分足を取得する
    """
    def __init__(self, stock_names, past_day=5, column_editor=ColumnNameEditorYahoo(), stop_time_span=2.0, is_use_stop=False, to_tokyo=True):
        """
        stock_names: str or list of str
            ロードしたい株の銘柄コード，東証なので.Tを付ける
        past_day: int
            ロードしたい日数，現在からこの日数だけロードされる
        column_editor: ColumnNameEditorYahoo
            カラム名調整のため．引数にする必要はなさそう
        stop_time_span: int
            apiを呼び出すときにストップする秒数．制限に引っかからないために設定する．とりあえず2秒にしておけば大丈夫
        is_use_stop: bool
            ストップするかどうか
        to_tokyo: bool
            タイムゾーンを東京へ変更するかどうか
        """
        if past_day >= 7:
            raise ValueError("yahoo finance cannot get more than 7 past day")
        
        self.stock_str_list_yahoo = ['open', 'high', 'low', 'close', 'volume']
        if isinstance(stock_names, str) and stock_names is not None:  # 一応一つだけのとき
            stock_names = [stock_names]
        self.stock_names = stock_names
        self.past_day = past_day
        self.column_editor = column_editor
        self.stop_time_span = stop_time_span
        self.is_use_stop = is_use_stop
        self.to_tokyo = to_tokyo
        
    def load(self):
        """
        データをロードする関数．
        returns
        -------
        left_df: pandas.DataFrame
            コンストラクタで指定した銘柄コードの始値Open，終値Close，高値High，安値Low，出来高VolumeをOpen_銘柄コードというカラム名で保持するDataFrameを返す．
        """
        IsFirst = True  # 最初かどうかのフラッグ
        IsMulti = False  # 複数かどうかのフラッグ
        
        if self.stock_names is None:
            raise ValueError("stock names is not setted")
        
        for stock_name in self.stock_names:
            stock_share = share.Share(stock_name)
            stock_data = None

            try:
                if self.is_use_stop:
                    YahooFinanceStockLoaderMin.api_now_used_time = time.time()  # apiのストップに利用する現在時間
                    if YahooFinanceStockLoaderMin.api_now_used_time is not None and YahooFinanceStockLoaderMin.api_last_used_time is not None:
                        api_span_time = YahooFinanceStockLoaderMin.api_now_used_time - YahooFinanceStockLoaderMin.api_last_used_time
                        if api_span_time < self.stop_time_span:
                            time.sleep(self.stop_time_span - api_span_time + 0.001)  # apiの制限をクリアするための停止
                    
                
                stock_data = stock_share.get_historical(share.PERIOD_TYPE_DAY,
                                                        self.past_day,
                                                        share.FREQUENCY_TYPE_MINUTE,
                                                        1
                                                        )
                
                YahooFinanceStockLoaderMin.api_last_used_time = time.time()  # apiを利用した最後の時間
                
            except YahooFinanceError as e:
                YahooFinanceStockLoaderMin.api_last_used_time = time.time()  # 一応エラーしたときも定めておく
                print(e.message)
                
            if stock_data is not None:
                df = pd.DataFrame()
                
                for stock_data_key in stock_data.keys():
                    if stock_data_key != "timestamp":  # timestampでないとき
                        df[self.column_editor(stock_name=stock_name, column_name=stock_data_key)] = stock_data[stock_data_key]  # dataframeへの代入
                        
                pandas_timestamp = [datetime.datetime.utcfromtimestamp(i/1000) for i in stock_data["timestamp"]]  # timestampをindexへ
                df.index = pd.DatetimeIndex(pandas_timestamp)
                df.index = df.index.tz_localize('UTC')
                if self.to_tokyo:   # tokyoのタイムゾーンに指定
                    df.index = df.index.tz_convert('Asia/Tokyo')
                
                if IsFirst:
                    left_df = df
                    IsFirst = False
                else:
                    IsMulti = True
                    right_df = df
                
                if IsMulti:  # 複数の場合
                    left_df = pd.merge(left_df, 
                                       right_df,
                                       how="outer",
                                       left_index=True,
                                       right_index=True
                                      )
        if IsFirst and not IsMulti:  # 一度もNoneが返ってこなかった場合
            return None
        
        # frequencyを定める
        left_df = left_df.asfreq("T")  # 分足データとして保持  
        left_df.index = left_df.index.set_names("timestamp")
        return left_df
        
    def set_stock_names(self, stock_names):
        """
        コンストラクタ以外で銘柄コードを設定する
        stock_names: str or list of str
            ロードしたい株の銘柄コード，なぜか.Tを付けないとうまく動かない        
        """
        if isinstance(stock_names, str):  # 一応一つだけのとき
            stock_names = [stock_names]
        self.stock_names = stock_names
        

if __name__ == "__main__":
    
    ##########################
    ### ColumnNameEditorYahoo
    ##########################

    stock_name = "4755.T"  # rakuten
    column_editor = ColumnNameEditorYahoo()
    yahoo_stock_str_list = ['open', 'high', 'low', 'close', 'volume']
    new_yahoo_stock_str_list = [column_editor(stock_name,i) for i in yahoo_stock_str_list]
    print(new_yahoo_stock_str_list)


    ##########################
    ### YahooFinanceStockLoaderMin
    ##########################

    stock_names = ["4755.T","6502.T"]

    stockloader = YahooFinanceStockLoaderMin(stock_names, stop_time_span=2.0, is_use_stop=False)
    stock_df = stockloader.load()
    print(stock_df.tail(5))
    
    