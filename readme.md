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