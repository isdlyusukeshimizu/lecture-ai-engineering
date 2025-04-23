# config.py
DB_FILE = "chat_feedback.db"
MODEL_NAME = "google/gemma-7b-it"
# MODEL_NAME = "google/gemma-1.1-2b-it"
# MODEL_NAME = "google/gemma-2-2b-jpn-it"

# 感想および考察
# gemma-7b-itはVRAMが必要と他の方の技術ブログに記載があったが、推論速度はあまり変わらなかった。
# 上記の考察として、Google Colab ProのA100/GPUを使っていたため、7Bモデルでも余裕で処理できる性能だったので、
# 2Bとあまり差が出なかったと考える。また、入力トークン数をかなり増やすことでパフォーマンスの変化を感じられた可能性もある。
