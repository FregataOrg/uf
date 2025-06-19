import os
import argparse
import base64
import json
import time
from io import BytesIO

from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# --- グローバル設定 ---
INPUT_DIR = "imgs"
OUTPUT_DIR = "outputs"
CACHE_DIR = "analysis_cache"
OLLAMA_MODEL = "llava"

# --- プロンプト定義 ---

# 1. ページ構造を解析させるためのプロンプト (パス1)
ANALYSIS_PROMPT = """
Analyze the layout and structure of this page from the 'Homotopy Type Theory' book.
Identify key components such as headers, footers, main text columns, figures, captions, and complex mathematical formula blocks.
Describe any potential challenges for OCR, such as multi-column layouts, rotated text, unusual fonts, or dense mathematical notation.
Output your analysis as a concise list of observations in JSON format. Example: {"observations": ["Two-column layout", "Contains a complex diagram in the bottom-right", "Header contains page number"]}
"""

# 2. 通常の抽出プロンプト (パス1)
BASIC_EXTRACTION_PROMPT = """
This image contains a page from the "Homotopy Type Theory: Univalent Foundations of Mathematics" book.
Please transcribe the content into a clean Markdown format.
Pay close attention to mathematical formulas and expressions, ensuring they are accurately represented, preferably using LaTeX syntax (e.g., $$ ... $$ or $ ... $).
Do not add any comments or explanations, only the transcribed Markdown content.
"""

# 3. 解析結果を付加して高精度な抽出を行うプロンプトのテンプレート (パス2)
REFINED_EXTRACTION_PROMPT_TEMPLATE = """
This image contains a page from the "Homotopy Type Theory" book.
A preliminary analysis of this page revealed the following structural points:
--- ANALYSIS ---
{analysis_text}
--- END ANALYSIS ---
Considering this analysis, please perform a highly accurate transcription of the content into a clean Markdown format.
Pay extra close attention to mathematical formulas, expressions, and the overall structure described in the analysis.
Ensure formulas are correctly represented using LaTeX syntax ($$...$$ or $...$).
Do not add any comments, just the final Markdown content.
"""


def image_to_base64(image_path: str) -> str:
    """画像をBase64文字列に変換する"""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"❌ 画像処理エラー: {image_path}, {e}")
        return None

def call_llm(chat_model, prompt_text, b64_image):
    """LLMを呼び出し、レスポンスを取得する"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"},
        ]
    )
    try:
        response = chat_model.invoke([message])
        return response.content
    except Exception as e:
        print(f"❌ Ollamaモデル呼び出しエラー: {e}")
        return None

def analyze_page_structure(chat_model, image_path, cache_path):
    """ページの構造を分析する (パス1)"""
    # キャッシュが存在すれば利用する
    if os.path.exists(cache_path):
        print(f"🧠 キャッシュ利用: {os.path.basename(cache_path)}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f).get("analysis_text")

    print(f"🔬 ページ構造を解析中: {os.path.basename(image_path)}")
    b64_image = image_to_base64(image_path)
    if not b64_image: return None

    analysis_text = call_llm(chat_model, ANALYSIS_PROMPT, b64_image)
    if analysis_text:
        # キャッシュに保存
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({"analysis_text": analysis_text}, f, ensure_ascii=False, indent=2)
        print(f"💾 解析結果を保存しました: {os.path.basename(cache_path)}")
    return analysis_text

def run_basic_process(chat_model, files_to_process):
    """指定されたファイル群に基本的な1パス処理を実行する"""
    print("\n--- 🚀 基本処理モードを開始します ---")
    for filename in files_to_process:
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".md"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_path):
            print(f"⏩ スキップ: {output_filename} は既に存在します。")
            continue

        print(f"📄 基本処理中: {filename}")
        b64_image = image_to_base64(input_path)
        if not b64_image: continue

        start_time = time.time()
        markdown_content = call_llm(chat_model, BASIC_EXTRACTION_PROMPT, b64_image)
        if markdown_content:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            processing_time = time.time() - start_time
            print(f"✅ 成功: {output_filename} を作成しました。({processing_time:.2f}秒)")

def run_refine_process(chat_model, refine_files):
    """指定されたファイル群に高精度な2パス処理を実行する"""
    print("\n--- ✨ 再処理（リファイン）モードを開始します ---")
    for filename in refine_files:
        print(f"\n--- ターゲット: {filename} ---")
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".md"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cache_filename = os.path.splitext(filename)[0] + ".json"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        # パス1: ページ構造の解析
        analysis_text = analyze_page_structure(chat_model, input_path, cache_path)
        if not analysis_text:
            print(f"❌ 解析失敗: {filename}")
            continue

        # パス2: 解析結果を利用した高精度抽出
        print(f"✍️ 高精度抽出を実行中: {filename}")
        b64_image = image_to_base64(input_path)
        if not b64_image: continue

        start_time = time.time()
        final_prompt = REFINED_EXTRACTION_PROMPT_TEMPLATE.format(analysis_text=analysis_text)
        markdown_content = call_llm(chat_model, final_prompt, b64_image)
        if markdown_content:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            processing_time = time.time() - start_time
            print(f"✅✅ 成功 (再処理): {output_filename} を更新しました。({processing_time:.2f}秒)")

def main():
    """メイン処理"""
    # ディレクトリ作成
    for dir_path in [OUTPUT_DIR, CACHE_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="HoTT BookのページをOCR処理し、Markdownに変換します。")
    parser.add_argument(
        '--refine',
        nargs='+',
        metavar='FILENAME',
        help="指定したファイル名に対して高精度な再処理（リファイン）を実行します。例: --refine page-005.jpg page-012.jpg"
    )
    args = parser.parse_args()

    print(f"🤖 Ollama ({OLLAMA_MODEL}モデル) を初期化します...")
    try:
        # temperatureを低めに設定し、より忠実な出力を促す
        chat = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    except Exception as e:
        print(f"❌ Ollamaモデルの初期化に失敗しました。詳細: {e}")
        return

    if args.refine:
        run_refine_process(chat, args.refine)
    else:
        try:
            all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.startswith("page-") and f.lower().endswith(".jpg")])
            run_basic_process(chat, all_files)
        except FileNotFoundError:
            print(f"❌ エラー: 入力ディレクトリ '{INPUT_DIR}' が見つかりません。")
            return

    print("\n--- すべての処理が完了しました ---")


if __name__ == "__main__":
    main()
