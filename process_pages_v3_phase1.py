import os
import json
import base64
import time
from io import BytesIO
from PIL import Image, ImageDraw

# LangChainのコンポーネントをインポート
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# --- グローバル設定 ---
# 注: お使いのQwen-VLモデル名に合わせて変更してください
OLLAMA_MODEL = "qwen-vl" 

# --- ディレクトリ設定 ---
INPUT_DIR = "imgs"
OUTPUT_DIR = "output"
LAYOUT_CACHE_DIR = "layout_cache"
TEMP_FIGURES_DIR = "temp_figures"

# --- プロンプト定義 ---

# 1. ページ全体のレイアウトを解析させるためのプロンプト
LAYOUT_ANALYSIS_PROMPT = """
You are an expert document layout analyzer. Your task is to analyze the provided image of a page from a scientific book.
Identify all distinct logical components on the page. For each component, provide its type and its precise bounding box coordinates `[x_min, y_min, x_max, y_max]`.
Also, determine the primary language of the document.

The component types should be one of the following:
- `text_block`: A standard paragraph of text.
- `formula_block`: A block containing one or more mathematical formulas, usually displayed on its own lines.
- `figure`: An image, diagram, or chart.
- `caption`: The text description for a figure.
- `header`: The page header (e.g., page number, chapter title).
- `footer`: The page footer.

Your output MUST be a single, valid JSON object following this exact schema. Do not add any other text or explanations.

JSON_OUTPUT_EXAMPLE:
```json
{
  "language": "english",
  "components": [
    {
      "type": "header",
      "box": [50, 50, 950, 100]
    },
    {
      "type": "text_block",
      "box": [100, 120, 900, 500]
    },
    {
      "type": "figure",
      "box": [200, 520, 800, 800]
    }
  ]
}
```
"""

# 2. トリミングされた個別のコンポーネントをOCRするためのプロンプト
COMPONENT_OCR_PROMPT = """
You are a highly specialized OCR engine. You will be given a small, pre-cropped image of a single document component.
Transcribe the content of this image with maximum accuracy.
- For text, output the plain text.
- For mathematical formulas, use proper LaTeX syntax.
Your output should be ONLY the transcribed content, with no extra explanations.
"""

# --- ヘルパー関数 ---

def setup_directories():
    """必要なディレクトリが存在しない場合に作成する"""
    for dir_path in [OUTPUT_DIR, LAYOUT_CACHE_DIR, TEMP_FIGURES_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def image_to_base64(pil_image: Image.Image) -> str:
    """Pillow画像をBase64文字列に変換する"""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_vllm(vllm_model, prompt_text, b64_image):
    """VLLMを呼び出し、レスポンスを取得する"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"},
        ]
    )
    try:
        response = vllm_model.invoke([message])
        # ```json ... ``` のようなマークダウンブロックからJSONを抽出する
        if "```json" in response.content:
            return response.content.split("```json")[1].split("```")[0].strip()
        return response.content
    except Exception as e:
        print(f"❌ VLLMモデル呼び出しエラー: {e}")
        return None

def get_reading_direction(language: str) -> str:
    """言語から読み取り方向のタグを決定する"""
    if language.lower() in ["japanese", "chinese"]:
        # ここではHoTT Bookが英語であると仮定しているが、将来的な拡張性のため
        return "TBRL" # Top-to-Bottom, Right-to-Left
    return "LRTB" # Left-to-Right, Top-to-Bottom

# --- メイン処理関数 ---

def analyze_and_extract_components(vllm_model, image_path: str):
    """
    【フェーズ1のメイン関数】
    画像を解析し、各コンポーネントを個別処理して構造化データを返す
    """
    filename = os.path.basename(image_path)
    base_filename = os.path.splitext(filename)[0]
    layout_cache_path = os.path.join(LAYOUT_CACHE_DIR, f"{base_filename}_layout.json")

    try:
        full_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 入力ファイルが見つかりません: {image_path}")
        return None
    
    # --- 1. レイアウト解析 ---
    layout_data = None
    if os.path.exists(layout_cache_path):
        print(f"🧠 キャッシュ利用: {filename} のレイアウト情報を読み込みます。")
        with open(layout_cache_path, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
    else:
        print(f"🔬 1. レイアウト解析を開始します: {filename}")
        full_image_b64 = image_to_base64(full_image)
        json_str = call_vllm(vllm_model, LAYOUT_ANALYSIS_PROMPT, full_image_b64)
        if json_str:
            try:
                layout_data = json.loads(json_str)
                with open(layout_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(layout_data, f, indent=2)
                print(f"💾 レイアウト情報をキャッシュに保存しました。")
            except json.JSONDecodeError as e:
                print(f"❌ レイアウト解析JSONのパースに失敗しました: {e}")
                print(f"   受信した文字列: {json_str}")
                return None
    
    if not layout_data or 'components' not in layout_data:
        print("❌ レイアウトデータが不正です。")
        return None

    print(f"✅ レイアウト解析完了。{len(layout_data['components'])}個のコンポーネントを検出。")

    # --- 2. 個別コンポーネント処理 ---
    print(f"\n🔬 2. 個別コンポーネントの処理を開始します...")
    processed_components = []
    for idx, comp in enumerate(layout_data['components']):
        comp_id = f"comp_{idx+1:02d}"
        comp_type = comp.get('type', 'unknown')
        box = tuple(comp.get('box', []))

        if not box or len(box) != 4:
            print(f"  -> スキップ: {comp_id} の座標が不正です。")
            continue
        
        print(f"  -> 処理中: {comp_id} (タイプ: {comp_type})")
        
        cropped_img = full_image.crop(box)
        
        content = ""
        if comp_type == 'figure':
            # 図は画像ファイルとして保存し、Markdownリンクを生成
            temp_fig_dir = os.path.join(TEMP_FIGURES_DIR, base_filename)
            os.makedirs(temp_fig_dir, exist_ok=True)
            temp_fig_path = os.path.join(temp_fig_dir, f"{comp_id}.jpg")
            cropped_img.save(temp_fig_path)
            content = f"![{comp_type} {comp_id}]({os.path.relpath(temp_fig_path)})"
        else:
            # テキストや数式などはOCRを実行
            cropped_img_b64 = image_to_base64(cropped_img)
            content = call_vllm(vllm_model, COMPONENT_OCR_PROMPT, cropped_img_b64)
            if not content:
                content = "[OCR FAILED]"

        processed_components.append({
            "id": comp_id,
            "type": comp_type,
            "box": list(box),
            "content": content
        })
        time.sleep(1) # APIへの連続アクセスを避けるための短い待機

    # --- 3. 最終的な構造化JSONの組み立て ---
    language = layout_data.get('language', 'unknown')
    final_output = {
        "source_file": filename,
        "language": language,
        "reading_direction": get_reading_direction(language),
        "components": processed_components
    }
    
    print("\n✅ フェーズ1の全処理が完了しました。")
    return final_output


if __name__ == '__main__':
    setup_directories()
    
    # --- テスト実行 ---
    # `imgs`ディレクトリにテストしたい画像を置いてください
    TEST_IMAGE_FILENAME = "page-003.jpg"
    test_image_path = os.path.join(INPUT_DIR, TEST_IMAGE_FILENAME)
    
    print("🤖 VLLMを初期化します...")
    try:
        # Temperatureを低く設定して、より忠実な出力を促す
        vllm = ChatOllama(model=OLLAMA_MODEL, temperature=0.05)
    except Exception as e:
        print(f"❌ VLLMの初期化に失敗しました: {e}")
        exit()
        
    # メイン関数を実行
    structured_data = analyze_and_extract_components(vllm, test_image_path)
    
    if structured_data:
        # 結果をJSONファイルとして保存
        output_filename = os.path.splitext(TEST_IMAGE_FILENAME)[0] + "_structured.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 最終的な構造化データを {output_path} に保存しました。")
        # print("\n--- 出力データ ---")
        # print(json.dumps(structured_data, indent=2, ensure_ascii=False))

