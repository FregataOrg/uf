import os
import base64
from io import BytesIO
from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import time

# --- 設定項目 ---

# 入力ディレクトリ
INPUT_DIR = "trim_imgs"

# 出力ディレクトリ
OUTPUT_DIR = "outputs"

# 使用するOllamaモデル名 (事前に `ollama run llava` を実行)
OLLAMA_MODEL = "z-uo/qwen2.5vl_tools:7b"

# LLMへの指示（プロンプト）
# HoTT Bookのページであることを伝え、数式に注意してMarkdownに変換するよう英語で指示
PROMPT_TEXT = """This image contains a page from the "Homotopy Type Theory: Univalent Foundations of Mathematics" book.
Please transcribe the content into a clean Markdown format.
Pay close attention to mathematical formulas and expressions, ensuring they are accurately represented, preferably using LaTeX syntax (e.g., $$ ... $$ or $ ... $).
Do not add any comments or explanations, only the transcribed Markdown content.
"""

# --- 関数定義 ---

def image_to_base64(image_path: str) -> str:
    """
    画像ファイルを読み込み、Base64エンコードされた文字列に変換する。
    """
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            # RGBA (透過) やパレットモードをRGBに変換
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(buffered, format="JPEG")
            img_byte = buffered.getvalue()
            return base64.b64encode(img_byte).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ エラー: 画像ファイルが見つかりません: {image_path}")
        return None
    except Exception as e:
        print(f"❌ エラー: 画像処理中に問題が発生しました: {image_path}, 詳細: {e}")
        return None

def process_single_image(chat_model: ChatOllama, input_path: str, output_path: str):
    """
    単一の画像を処理し、結果をMarkdownファイルとして保存する。
    """
    print(f"📄 処理中: {os.path.basename(input_path)}")

    # 画像をBase64に変換
    base64_image = image_to_base64(input_path)
    if not base64_image:
        return

    # LLMへの入力メッセージを作成
    message = HumanMessage(
        content=[
            {"type": "text", "text": PROMPT_TEXT},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
        ]
    )

    # モデルを呼び出し、結果を取得
    try:
        start_time = time.time()
        response = chat_model.invoke([message])
        end_time = time.time()
        
        # 結果をファイルに書き込み
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.content)
            
        processing_time = end_time - start_time
        print(f"✅ 成功: {os.path.basename(output_path)} を作成しました。(処理時間: {processing_time:.2f}秒)")

    except Exception as e:
        print(f"❌ エラー: Ollamaモデルの呼び出し中にエラーが発生しました。({os.path.basename(input_path)})")
        print(f"   Ollamaが起動しているか確認してください。詳細: {e}")

def main():
    """
    メイン処理
    """
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(OUTPUT_DIR):
        print(f"📁 出力ディレクトリを作成します: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    # 入力ディレクトリ内の画像ファイルを取得
    try:
        image_files = sorted([
            f for f in os.listdir(INPUT_DIR) 
            if f.startswith("page-") and f.lower().endswith(".jpg")
        ])
    except FileNotFoundError:
        print(f"❌ エラー: 入力ディレクトリが見つかりません: {INPUT_DIR}")
        print("   'imgs' ディレクトリを作成し、'page-XXX.jpg' 形式の画像ファイルを入れてください。")
        return

    if not image_files:
        print(f"🔍 {INPUT_DIR} 内に処理対象の画像ファイルが見つかりませんでした。")
        return

    print(f"🤖 Ollama ({OLLAMA_MODEL}モデル) を初期化します...")
    try:
        # temperatureを低めに設定し、より忠実な出力を促す
        chat = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    except Exception as e:
        print(f"❌ エラー: Ollamaモデルの初期化に失敗しました。詳細: {e}")
        return
        
    print(f"--- 処理を開始します (対象ファイル数: {len(image_files)}) ---")

    for filename in image_files:
        input_path = os.path.join(INPUT_DIR, filename)
        
        # 出力ファイルパスを生成 (例: page-001.jpg -> page-001.md)
        output_filename = os.path.splitext(filename)[0] + ".md"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # 既にファイルが存在する場合はスキップ
        if os.path.exists(output_path):
            print(f"⏩ スキップ: {os.path.basename(output_path)} は既に存在します。")
            continue

        process_single_image(chat, input_path, output_path)

    print("\n--- すべての処理が完了しました ---")


if __name__ == "__main__":
    main()