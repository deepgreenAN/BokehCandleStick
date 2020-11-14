import pandas as pd
import datetime
from datetime import timedelta
import re
import warnings
from pytz import timezone

# 自分が利用する足
available_sample_type = set(["S", "1S", "30S", "T", "1T", "5T", "10T", "30T", "H","1H", "2H", "4H", 
                        "12H", "D", "1D", "B", "1B", "W", "1W", "2W", "M", "1M", "Q", "1Q", "Y", "1Y"])

middle_type_dict ={"min":"T","1min":"T","T":"T","1T":"T","5T":"5T","10T":"10T","30T":"30T",
                   "H":"H","1H":"H","2H":"2H","4H":"4H","12H":"12H","D":"D","1D":"D",
                   "B":"B","1B":"B","W":"W","1W":"W","2W":"2W","M":"M","1M":"M","Q":"Q","1Q":"Q",
                   "Y":"Y","1Y":"Y"
                   }

# frequencyとtimedeltaの辞書Sから2Wまで
type_deltataime_dict = {
    "S":timedelta(seconds=1),"30S":timedelta(seconds=30),"T":timedelta(minutes=1),"5T":timedelta(minutes=5),"10T":timedelta(minutes=10),
    "30T":timedelta(minutes=30),"H":timedelta(hours=1),"2H":timedelta(hours=2),"4H":timedelta(hours=4),"12H":timedelta(hours=12),
    "D":timedelta(days=1),"W":timedelta(days=7),"2W":timedelta(days=14)
}

seconds_freq = {type_deltataime_dict[i].total_seconds():i for i in type_deltataime_dict.keys()}

freq_seconds = {i:type_deltataime_dict[i].total_seconds() for i in type_deltataime_dict.keys()}

def middle_sample_type_with_check(freq):
    """
    与えられたfreq_strが使えるものかどうか判断し，pandasで利用可能なfreq_strを返す．
    freq: str
        frequencyを意味する文字列
    """
    if freq not in available_sample_type:
        raise ValueError("This freq is invalid")

    return middle_type_dict[freq]

def get_freq_from_sec(seconds):
    """
    指定した秒数から，pandasで指定されるfrequency strを出力する．ただし，"S"から"2W"まで
    seconds: int
        frequencyを求めたい整数
    """
    #import pdb; pdb.set_trace()  # get_freq_from_sec
    if seconds not in seconds_freq:
        raise ValueError("This seconds is invalid")

    return seconds_freq[seconds]

def get_sec_from_freq(freq):
    """
    指定したfrequency strから，秒数を出力する関数．ただし"S"から"2W"まで
    freq: str
        秒数を求めたいfrequency str
    """
    #import pdb; pdb.set_trace()  # get_sec_from_freq
    if freq not in freq_seconds:
        raise ValueError("This freq is invalid")
    return freq_seconds[freq]


def get_df_freq(df):
    """
    pandasのdfのfreqを出力する関数
    """

    if df.index.freqstr is not None:
        return df.index.freqstr
        
    if len(df.index) < 2:
        raise ValueError("cannot caolculate frequency of dataframe")
        
    first_datetime = df.index[0].to_pydatetime()  # datetime.datetimeに変換
    second_datetime = df.index[1].to_pydatetime()  # datetime.datetimeに変換

    diff_seconds = second_datetime.timestamp() - first_datetime.timestamp()  # unixに変換
    return get_freq_from_sec(diff_seconds)


def get_next_datetime(select_datetime, freq_str, has_mod=True):
        """
        与えられた日時から，frequency_strを考慮して次の時間を求めるための関数．余りが出てしまった場合， warningを出す．
        "D"以下，さらに"W","2W","M","Y"を前提としている．
        select_datetime: datetime.datetime(aware or naive)
            与える日時，awareなdatetimeかlocal timeを前提としたnaiveなdatetime
        freq_str: str
            frequencyに対応する文字列
        has_first: bool
            与える日時に余りがあるかどうか，
        """
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        freq_str = middle_sample_type_with_check(freq_str)

        select_is_aware = select_datetime.tzinfo is not None  # select_datetimeがawareかどうか
        if select_is_aware:
            select_timezone = timezone(str(select_datetime.tzinfo))

        if has_mod:  #（余りを含む可能性がある
            if freq_str in {"W","2W"}:  # 週足あるいは2週足を求めるとき
                years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
                hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

                weekday = select_datetime.weekday()
                if {hours, minutes, seconds} != {0} or weekday != 6:  # 時間以下が全て0でないときあるいは，日曜でなかった場合
                    warnings.warn("input time is not coresponded to frequency. resampling result may be wrong")
                    if {hours, minutes, seconds} != {0}:
                        hours, minutes, seconds = 0,0,0
        
                    weekday += 1  # 月曜始まりを日曜始まりに強引に変更
                    if freq_str == "W":  # 週足の場合
                        add_days = 7 - weekday
                    elif freq_str == "2W":  # 2週足の場合
                        add_days = 14 - weekday
                else:  # 日曜だった場合
                    if freq_str == "W":  # 週足の場合
                        add_days= 7
                    elif freq_str == "2W":  # 2週足の場合
                        add_days = 14
                
                next_datetime = datetime.datetime(years, months, days+add_days, hours, minutes, seconds)
                if select_is_aware:  # select_datetmeがawareな場合
                    next_datetime = select_timezone.localize(next_datetime)
                
            elif freq_str == "M":  # 月足を求めるとき
                years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
                hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

                if {days, hours, minutes, seconds} != {0}:  # 日にち以下がすべて0でないとき(月の開始時でないとき)
                    warnings.warn("input time is not coresponded to frequency. next datetime may be wrong")
                    days, hours, minutes, seconds = 1,0,0,0

                if months == 12:  # 12月のとき
                    years += 1
                    months = 1  # １月にする
                else:  # 月の開始時のとき
                    months += 1  # 月を一つ加える
                next_datetime = datetime.datetime(years, months, days, hours, minutes, seconds)
                if select_is_aware:  # select_datetmeがawareな場合
                    next_datetime = select_timezone.localize(next_datetime)

            elif freq_str == "Y":  # 年足で求めるとき（元旦始まり）
                years, months, days = select_datetime.year, select_datetime.month, select_datetime.day
                hours, minutes, seconds = select_datetime.hour, select_datetime.minute, select_datetime.second

                if {months, days, hours, minutes, seconds} != {0}:  # 月以下がすべて0でないとき
                    warnings.warn("input time is not coresponded to frequency. next datetime may be wrong")
                    months, days, hours, minutes, seconds = 1,1,0,0,0
                
                years += 1
                next_datetime = datetime.datetime(years, months, days, hours, minutes, seconds)
                if select_is_aware:  # select_datetmeがawareな場合
                    next_datetime = select_timezone.localize(next_datetime)

            else:  # それ以下の足を求める場合
                sec_from_freq = get_sec_from_freq(freq_str)
                if select_is_aware:
                    timezone_datetime = select_datetime
                else:
                    utc_timezone = timezone("UTC")  # これはどこでもよい
                    timezone_datetime = utc_timezone.localize(select_datetime)
                # select_datetimeと同じdatetimeの値を持つutcのunix時間＋sec_from_freqを取得
                next_time_unix = timezone_datetime.timestamp() + timezone_datetime.utcoffset().total_seconds() + sec_from_freq  # unix時間
                
                next_time_mod = int(next_time_unix) % int(sec_from_freq)
                if next_time_mod != 0:  # 余りが出てしまった場合
                    warnings.warn("input time is not coresponded to frequency. next datetime may be wrong")
                    next_time_unix = next_time_unix - next_time_mod
                    
                next_datetime = datetime.datetime.utcfromtimestamp(next_time_unix)  # utcのnaiveなdatetimeになる(datetimeの値に準ずる)
                if select_is_aware:  # select_datetmeがawareな場合
                    next_datetime = select_timezone.localize(next_datetime)

        else:  # （余りを含んでいないとき）
            if freq_str in {"W","2W"}:  # 週足あるいは2週足を求めるとき
                if freq_str == "W":  # 週足を求める場合
                    next_datetime = select_datetime + timedelta(days=7)
                elif freq_str == "2W":  # 2週足を求める場合
                    next_datetime = select_datetime + timedelta(days=14)

            elif freq_str == "M":  # 月足の場合
                years, months = select_datetime.year, select_datetime.month
                if months == 12:  # 12月のとき
                    years += 1
                    months = 1  # １月にする
                else:
                    months += 1  # 月を一つ加える
                next_datetime = datetime.datetime(years, months, 1, 0, 0, 0)
                if select_is_aware:  # select_datetmeがawareな場合
                    next_datetime = select_timezone.localize(next_datetime)

            elif freq_str == "Y":  # 年足の場合，元旦始まり
                years = select_datetime.year
                years += 1
                next_datetime = datetime.datetime(years, 1, 1, 0, 0, 0)
                if select_is_aware:  # select_datetmeがawareな場合
                    next_datetime = select_timezone.localize(next_datetime)

            else:  # それ以下の足の場合
                sec_from_freq = get_sec_from_freq(freq_str)
                next_datetime = select_datetime + timedelta(seconds=sec_from_freq)
        
        return next_datetime

def get_utc_naive_datetime_from_datetime(select_datetime):
    """
    awareなdatetimeかnaiveなdatetime(local timeを前提)をutcのnaiveなdatetimeに返る
    select_datetime: datetime
    """
    # awareなdatetimeからutc
    select_datetime_unix = select_datetime.timestamp()  #unix時刻
    utc_naive_datetime = datetime.datetime.utcfromtimestamp(select_datetime_unix)
    return utc_naive_datetime


# 日本株の場合
#base_ohlc_patterns = {
#    "Open":re.compile("Open_[0-9]+"),
#    "High":re.compile("High_[0-9]+"),
#    "Low":re.compile("Low_[0-9]+"),
#    "Close":re.compile("Close_[0-9]+"),
#    "Volume":re.compile("Volume_[0-9]+")
#}

# アメリカ株を含む
base_ohlc_patterns = {
    "Open":re.compile("Open_.+"),
    "High":re.compile("High_.+"),
    "Low":re.compile("Low_.+"),
    "Close":re.compile("Close_.+"),
    "Volume":re.compile("Volume_.+")
}

class ConvertFreqOHLCV():
    """
    dataframeの足をダウンサンプリングする．
    """
    def __init__(self, freq_str, ohlcv_patterns=base_ohlc_patterns):
        """
        freq_str: str
            サンプリング周期に対応する文字列
        ohlcv_patterns: dict of str:pattern
            Open,High,Low,Close,Volumeとそれに対応するパターン
        """
        freq_str = middle_sample_type_with_check(freq_str)
        self.freq_str = freq_str
        
        self.ohlcv_patterns = ohlcv_patterns
         
    def __call__(self, df):
        # dfのfrequencyの判定とアップサンプリング時のエラー
        df_freq = get_df_freq(df)
        df_freq = middle_sample_type_with_check(df_freq)
        if get_sec_from_freq(df_freq) > get_sec_from_freq(self.freq_str):  # アップサンプリングの時
            raise ValueError("this is upsampling")
        
        # ohlcvそれぞれ対応するカラムを求める(aggに渡す辞書を求める)
        column_series = pd.Series(len(df.columns)*[None], index=df.columns)

        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["Open"])] = "first"
        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["High"])] = "max"
        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["Low"])] = "min"
        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["Close"])] = "last"
        column_series.loc[column_series.index.str.match(self.ohlcv_patterns["Volume"])] = "sum"
        agg_dict = dict(column_series)

        resampled_df = df.resample(self.freq_str).agg(agg_dict)  # resampling
        return resampled_df

if __name__ == "__main__":
    from get_stock_price import YahooFinanceStockLoaderMin
    
    ##########################
    ### get_df_freq
    ##########################
    
    stock_names = ["4755.T"]

    stockloader = YahooFinanceStockLoaderMin(stock_names, stop_time_span=2.0, is_use_stop=False)
    stock_df = stockloader.load()
    print(get_df_freq(stock_df))

    datetime_list = [datetime.datetime(2020, 10, 25, 12, 0, 0),
                     datetime.datetime(2020, 10, 25, 12, 1, 0),
                     datetime.datetime(2020, 10, 25, 12, 2, 0)
                    ]
    df = pd.DataFrame([[0,1],[2,3],[3,4]], columns=["A","B"],index=pd.DatetimeIndex(datetime_list))
    print(df.index.freqstr)
    print(get_df_freq(df))

    freq_converter = ConvertFreqOHLCV("5T")
    resampled_df = freq_converter(stock_df)
    print(resampled_df.tail(5))

    ##########################
    ### get_next_datetime
    ##########################
    
    select_datetime = datetime.datetime(2020, 11, 4, 12, 30, 0)
    print("naive select_datetime:",select_datetime)
    freq_str = "H"
    next_datetime = get_next_datetime(select_datetime, freq_str, has_mod=True)
    

    jst_timezone = timezone("Asia/Tokyo")
    select_datetime = jst_timezone.localize(datetime.datetime(2020, 11, 4, 12, 30, 0))
    print("naive select_datetime:",select_datetime)
    freq_str = "4H"
    next_datetime = get_next_datetime(select_datetime, freq_str, has_mod=True)
    print("next time:",next_datetime)