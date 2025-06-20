import os
import json
import base64
import time
from io import BytesIO
from PIL import Image

# LangChainのコンポーネントをインポート
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# --- グローバル設定 ---
# フェーズ1用VLLM (視覚と言語)
VLLM_MODEL = "qwen-vl" 
# フェーズ2用高次LLM (論理的推論)
# ★★★ ここに再結合用の高性能モデルを指定してください (例: 'llama3:70b', 'qwen2:72b') ★★★
REASSEMBLY_LLM_MODEL = "qwen2"

# --- ディレクトリ設定 ---
INPUT_DIR = "imgs"
OUTPUT_DIR = "output"
LAYOUT_CACHE_DIR = "layout_cache"
TEMP_FIGURES_DIR = "temp_figures"

# --- プロンプト定義 ---

# (フェーズ1) 1. ページ全体のレイアウトを解析させるためのプロンプト
LAYOUT_ANALYSIS_PROMPT = """
You are an expert document layout analyzer. Your task is to analyze the provided image of a page from a scientific book.
Identify all distinct logical components on the page. For each component, provide its type and its precise bounding box coordinates `[x_min, y_min, x_max, y_max]`.
Also, determine the primary language of the document.
The component types should be one of the following: `text_block`, `formula_block`, `figure`, `caption`, `header`, `footer`.
Your output MUST be a single, valid JSON object following this exact schema. Do not add any other text or explanations.
JSON_OUTPUT_EXAMPLE:
```json
{
  "language": "english",
  "components": [
    {"type": "header", "box": [50, 50, 950, 100]},
    {"type": "text_block", "box": [100, 120, 900, 500]},
    {"type": "figure", "box": [200, 520, 800, 800]}
  ]
}
```
"""

# (フェーズ1) 2. トリミングされた個別のコンポーネントをOCRするためのプロンプト
COMPONENT_OCR_PROMPT = """
You are a highly specialized OCR engine. You will be given a small, pre-cropped image of a single document component.
Transcribe the content of this image with maximum accuracy.
- For text, output the plain text.
- For mathematical formulas, use proper LaTeX syntax.
Your output should be ONLY the transcribed content, with no extra explanations.
"""

# (フェーズ2) 3. バラバラのコンポーネントを論理的に再結合させるためのプロンプト
REASSEMBLY_PROMPT_TEMPLATE = """
You are an expert technical editor. Your task is to reconstruct a page from a scientific document using a list of disordered content components.

### CONTEXT
- The document language is: **{language}**
- The standard reading direction is: **{reading_direction}** (This is a general rule, but logical connection is more important).

### INPUT DATA (List of Document Components)
Here are all the components extracted from the page in JSON format. Each component has an ID, type, bounding box (a hint for position), and its transcribed content.
```json
{components_json}
```

### YOUR TASK
1.  Analyze the `content` of all components to understand their logical relationships.
2.  Determine the most natural reading order that forms a coherent, logical flow.
3.  **PRIORITIZE SEMANTIC CONNECTION OVER POSITION.** For example, a sentence fragment in one block should be followed by its continuation in another, regardless of their `box` coordinates. A figure's caption should follow the figure or its reference.
4.  Your final output MUST be a single JSON array containing the component `id`s in the correct logical order.

### OUTPUT FORMAT EXAMPLE
`["comp_01", "comp_03", "comp_02", "comp_05", "comp_04"]`
"""

# --- ヘルパー関数 (変更なし) ---
def setup_directories():
    for dir_path in [OUTPUT_DIR, LAYOUT_CACHE_DIR, TEMP_FIGURES_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def image_to_base64(pil_image: Image.Image) -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_llm(model, prompt_text, b64_image=None):
    """LLM/VLLMを呼び出し、レスポンスを取得する (画像はオプショナル)"""
    content_list = [{"type": "text", "text": prompt_text}]
    if b64_image:
        content_list.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"})
    
    message = HumanMessage(content=content_list)
    try:
        response = model.invoke([message])
        content = response.content
        # ```json ... ``` のようなマークダウンブロックからJSONを抽出する
        if "```json" in content:
            return content.split("```json")[1].split("```")[0].strip()
        # `[` で始まるJSON配列を直接探す
        if content.strip().startswith("["):
            return content.strip()
        return content
    except Exception as e:
        print(f"❌ LLM/VLLMモデル呼び出しエラー: {e}")
        return None

def get_reading_direction(language: str) -> str:
    if language.lower() in ["japanese", "chinese"]: return "TBRL"
    return "LRTB"

# --- フェーズ1: レイアウト解析と個別OCR (変更なし) ---
def analyze_and_extract_components(vllm_model, image_path: str):
    filename = os.path.basename(image_path)
    base_filename = os.path.splitext(filename)[0]
    layout_cache_path = os.path.join(LAYOUT_CACHE_DIR, f"{base_filename}_layout.json")
    try:
        full_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError: return None
    layout_data = None
    if os.path.exists(layout_cache_path):
        with open(layout_cache_path, 'r', encoding='utf-8') as f: layout_data = json.load(f)
    else:
        full_image_b64 = image_to_base64(full_image)
        json_str = call_llm(vllm_model, LAYOUT_ANALYSIS_PROMPT, full_image_b64)
        if json_str:
            try:
                layout_data = json.loads(json_str)
                with open(layout_cache_path, 'w', encoding='utf-8') as f: json.dump(layout_data, f, indent=2)
            except json.JSONDecodeError: return None
    if not layout_data or 'components' not in layout_data: return None
    
    processed_components = []
    for idx, comp in enumerate(layout_data['components']):
        comp_id = f"comp_{idx+1:02d}"; comp_type = comp.get('type', 'unknown'); box = tuple(comp.get('box', []))
        if not box or len(box) != 4: continue
        cropped_img = full_image.crop(box)
        content = ""
        if comp_type == 'figure':
            temp_fig_dir = os.path.join(TEMP_FIGURES_DIR, base_filename)
            os.makedirs(temp_fig_dir, exist_ok=True)
            temp_fig_path = os.path.join(temp_fig_dir, f"{comp_id}.jpg")
            cropped_img.save(temp_fig_path); content = f"![{comp_type} {comp_id}]({os.path.relpath(temp_fig_path)})"
        else:
            content = call_llm(vllm_model, COMPONENT_OCR_PROMPT, image_to_base64(cropped_img)) or "[OCR FAILED]"
        processed_components.append({"id": comp_id, "type": comp_type, "box": list(box), "content": content})
        time.sleep(0.5)
    
    language = layout_data.get('language', 'unknown')
    return {"source_file": filename, "language": language, "reading_direction": get_reading_direction(language), "components": processed_components}

# --- フェーズ2: インテリジェント再結合 (新規追加) ---
def reassemble_components(reassembly_llm, page_data: dict) -> str:
    """
    【フェーズ2のメイン関数】
    高次LLMを使ってコンポーネントを論理的に並べ替え、最終的なMarkdownを生成する
    """
    print("\n🔬 3. インテリジェント再結合を開始します...")
    
    # LLMに渡すために、content以外の不要なキーを一時的に削除したコンポーネントリストを作成
    components_for_prompt = [{"id": c["id"], "type": c["type"], "box": c["box"], "content": c["content"]} for c in page_data["components"]]
    
    # プロンプトを組み立て
    prompt = REASSEMBLY_PROMPT_TEMPLATE.format(
        language=page_data['language'],
        reading_direction=page_data['reading_direction'],
        components_json=json.dumps(components_for_prompt, indent=2)
    )
    
    # 高次LLMを呼び出し
    response_str = call_llm(reassembly_llm, prompt)
    
    if not response_str:
        print("❌ 再結合LLMからの応答がありません。")
        return None
    
    try:
        # LLMからの応答(IDのJSON配列)をパース
        ordered_ids = json.loads(response_str)
        print(f"✅ LLMによる順序決定: {ordered_ids}")
    except json.JSONDecodeError:
        print(f"❌ 再結合LLMの応答JSONのパースに失敗しました。")
        print(f"   受信した文字列: {response_str}")
        return None
    
    # 元のコンポーネントデータをIDで検索できるように辞書に変換
    components_map = {c['id']: c for c in page_data['components']}
    
    # 決定された順序でcontentを結合
    final_markdown_parts = []
    for comp_id in ordered_ids:
        if comp_id in components_map:
            final_markdown_parts.append(components_map[comp_id]['content'])
        else:
            print(f"⚠️ 警告: LLMが未知のID '{comp_id}' を返しました。")

    return "\n\n".join(final_markdown_parts)

if __name__ == '__main__':
    setup_directories()
    
    TEST_IMAGE_FILENAME = "page-003.jpg" # テストしたい画像ファイル名
    test_image_path = os.path.join(INPUT_DIR, TEST_IMAGE_FILENAME)
    base_filename = os.path.splitext(TEST_IMAGE_FILENAME)[0]
    
    print("🤖 モデルを初期化します...")
    try:
        vllm = ChatOllama(model=VLLM_MODEL, temperature=0.05)
        reassembly_llm = ChatOllama(model=REASSEMBLY_LLM_MODEL, temperature=0.0) # 再結合は決定的な方が良い
    except Exception as e:
        print(f"❌ モデルの初期化に失敗しました: {e}"); exit()
        
    # --- フェーズ1実行 ---
    structured_data = analyze_and_extract_components(vllm, test_image_path)
    if not structured_data:
        print("フェーズ1で処理が停止しました。"); exit()
        
    structured_data_path = os.path.join(OUTPUT_DIR, f"{base_filename}_structured.json")
    with open(structured_data_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ フェーズ1完了。中間データを {structured_data_path} に保存しました。")

    # --- フェーズ2実行 ---
    final_markdown = reassemble_components(reassembly_llm, structured_data)

    if final_markdown:
        final_md_path = os.path.join(OUTPUT_DIR, f"{base_filename}_final.md")
        with open(final_md_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        print(f"\n🎉 全工程完了！最終的なMarkdownを {final_md_path} に保存しました。")
    else:
        print("\n❌ フェーズ2で処理が失敗しました。")
