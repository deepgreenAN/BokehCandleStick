from pytz import timezone
import datetime
from bokeh.io import curdoc

from yahoo_stock_reader_ver1 import YahooFinanceStockLoaderMin
from bokeh_candle_stick import BokehCandleStickDf


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

curdoc().add_root(bokeh_candle_stick.dp)
curdoc().add_periodic_callback(bokeh_candle_stick.update, 1000)
