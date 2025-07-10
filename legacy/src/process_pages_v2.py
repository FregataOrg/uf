import os
import argparse
import base64
import json
import time
import re
from io import BytesIO

from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# --- グローバル設定 ---
INPUT_DIR = "trim_imgs"
OUTPUT_DIR = "outputs_v2"
CACHE_DIR = "analysis_cache"
OLLAMA_MODEL = "qwen2.5vl:32b"
# ★★★ 状態管理ファイル ★★★
STRUCTURE_FILE = "document_structure.json"


# --- プロンプトテンプレート定義 ---
# これらは後ほど動的にコンテキストを付加するための「元」となります

SYSTEM_PROMPT = """
You are an expert digital archivist specializing in mathematical and scientific texts. Your task is to perform high-fidelity Optical Character Recognition (OCR) and document layout analysis, converting physical pages into perfectly structured Markdown documents with accurate LaTeX formatting.
"""

PERFECT_EXAMPLE_MARKDOWN = """
...as shown in the equation:

$$ \sum_{i=0}^{n} i = \frac{n(n+1)}{2} $$

This is followed by more text.
"""

# --- 状態管理ヘルパー関数 ---

def load_document_state():
    """文書構造の状態ファイルを読み込む。なければ初期状態を返す。"""
    if not os.path.exists(STRUCTURE_FILE):
        print("📖 状態ファイルが見つからないため、新しい文書として開始します。")
        return {"last_processed_page": None, "heading_context_stack": []}
    try:
        with open(STRUCTURE_FILE, 'r', encoding='utf-8') as f:
            print(f"📖 状態ファイル '{STRUCTURE_FILE}' を読み込みました。")
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ 状態ファイルの読み込みに失敗しました: {e}。初期状態から開始します。")
        return {"last_processed_page": None, "heading_context_stack": []}

def save_document_state(state):
    """現在の文書構造の状態をファイルに保存する。"""
    try:
        with open(STRUCTURE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"❌ 状態ファイルの保存に失敗しました: {e}")

def update_heading_stack(stack, markdown_content):
    """生成されたMarkdownを解析し、見出しスタックを更新する。"""
    headings = re.findall(r'^(#+)\s+(.*)', markdown_content, re.MULTILINE)
    if not headings:
        return stack

    for heading_marks, title in headings:
        level = len(heading_marks)
        new_heading = {"level": level, "title": title.strip()}
        while stack and stack[-1]['level'] >= level:
            stack.pop()
        stack.append(new_heading)

    print(f"📚 見出しコンテキスト更新: 現在地 -> {format_context_for_prompt(stack)[0]}")
    return stack

def format_context_for_prompt(stack):
    """見出しスタックをLLM向けのプロンプト形式に変換する。"""
    if not stack:
        context_str = "No current context (This is likely the beginning of the document)."
        depth = 0
    else:
        path_str = " > ".join([f"{'#' * h['level']} {h['title']}" for h in stack])
        depth = stack[-1]['level']
        context_str = f"Current Path: {path_str}\nCurrent Heading Depth: {depth}"

    prompt_context = f"""
--- DOCUMENT CONTEXT ---
You are currently inside the following document structure. Use this information to determine the correct heading levels (e.g., if current depth is 2, a new major heading on the page is likely level 3 '###').
{context_str}
--- END DOCUMENT CONTEXT ---
"""
    return prompt_context, depth

# --- 既存のヘルパー関数 ---
def image_to_base64(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO(); img.convert('RGB').save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"❌ 画像処理エラー: {image_path}, {e}"); return None

def call_llm(chat_model, prompt_text, b64_image):
    message = HumanMessage(content=[{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"}])
    try:
        return chat_model.invoke([message]).content
    except Exception as e:
        print(f"❌ Ollamaモデル呼び出しエラー: {e}"); return None

def analyze_page_structure(chat_model, image_path, cache_path, context_prompt):
    if os.path.exists(cache_path):
        print(f"🧠 キャッシュ利用: {os.path.basename(cache_path)}")
        with open(cache_path, 'r', encoding='utf-8') as f: return json.load(f).get("analysis_text")
    
    analysis_prompt = f"{SYSTEM_PROMPT}\n{context_prompt}\nAnalyze the layout... (以下、元のANALYSIS_PROMPTと同様)" # 簡略化
    print(f"🔬 ページ構造を解析中: {os.path.basename(image_path)}")
    b64_image = image_to_base64(image_path)
    if not b64_image: return None
    analysis_text = call_llm(chat_model, analysis_prompt, b64_image)
    if analysis_text:
        with open(cache_path, 'w', encoding='utf-8') as f: json.dump({"analysis_text": analysis_text}, f, ensure_ascii=False, indent=2)
        print(f"💾 解析結果を保存しました: {os.path.basename(cache_path)}")
    return analysis_text

# --- 主要処理関数 (コンテキスト対応版) ---

def run_basic_process(chat_model, files_to_process, state):
    print("\n--- 🚀 基本処理モードを開始します ---")
    try:
        start_index = files_to_process.index(state['last_processed_page']) + 1 if state['last_processed_page'] in files_to_process else 0
    except ValueError: start_index = 0
    if start_index > 0: print(f"▶️ 前回の処理 '{state['last_processed_page']}' の次から再開します。")

    files_to_run = files_to_process[start_index:]
    if not files_to_run: print("✅ すべてのファイルは既に処理済みのようです。"); return

    for filename in files_to_run:
        print(f"\n--- ターゲット: {filename} ---")
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".md")

        context_prompt_str, _ = format_context_for_prompt(state['heading_context_stack'])
        basic_prompt = f"{SYSTEM_PROMPT}\n{context_prompt_str}\nYour task is to transcribe the provided page... (以下、元のBASIC_EXTRACTION_PROMPTと同様)" # 簡略化
        
        print(f"📄 基本処理中: {filename}")
        b64_image = image_to_base64(input_path)
        if not b64_image: continue

        start_time = time.time()
        markdown_content = call_llm(chat_model, basic_prompt, b64_image)
        
        if markdown_content:
            with open(output_path, "w", encoding="utf-8") as f: f.write(markdown_content)
            print(f"✅ 成功: {os.path.basename(output_path)} を作成しました。({time.time() - start_time:.2f}秒)")
            state['heading_context_stack'] = update_heading_stack(state['heading_context_stack'], markdown_content)
            state['last_processed_page'] = filename
            save_document_state(state)
        else:
            print(f"❌ 処理失敗: {filename}。処理を中断します。"); break

def run_refine_process(chat_model, refine_files, state):
    print("\n--- ✨ 再処理（リファイン）モードを開始します ---")
    print("⚠️ 注意: リファインモードは文書全体の状態を更新しません。")
    context_prompt_str, _ = format_context_for_prompt(state['heading_context_stack'])

    for filename in refine_files:
        print(f"\n--- ターゲット: {filename} ---")
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".md")
        cache_path = os.path.join(CACHE_DIR, os.path.splitext(filename)[0] + ".json")

        analysis_text = analyze_page_structure(chat_model, input_path, cache_path, context_prompt_str)
        if not analysis_text: print(f"❌ 解析失敗: {filename}"); continue

        print(f"✍️ 高精度抽出を実行中: {filename}")
        b64_image = image_to_base64(input_path)
        if not b64_image: continue

        # REFINED_EXTRACTION_PROMPT_TEMPLATEの元々の内容を再現しつつコンテキストを挿入
        refined_template = f"""{SYSTEM_PROMPT}
{context_prompt_str}
You will perform a highly accurate transcription of the provided page.
First, study this example:
--- EXAMPLE START ---
{PERFECT_EXAMPLE_MARKDOWN}
--- END EXAMPLE ---
Next, study this analysis:
--- ANALYSIS ---
{{analysis_text}}
--- END ANALYSIS ---
Considering both, transcribe the entire page.
"""
        final_prompt = refined_template.replace('{analysis_text}', analysis_text)
        
        start_time = time.time()
        markdown_content = call_llm(chat_model, final_prompt, b64_image)
        if markdown_content:
            with open(output_path, "w", encoding="utf-8") as f: f.write(markdown_content)
            print(f"✅✅ 成功 (再処理): {os.path.basename(output_path)} を更新しました。({time.time() - start_time:.2f}秒)")

def main():
    for dir_path in [OUTPUT_DIR, CACHE_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        
    document_state = load_document_state()

    parser = argparse.ArgumentParser(description="HoTT BookのページをOCR処理し、Markdownに変換します。")
    parser.add_argument('--refine', nargs='+', metavar='FILENAME', help="指定したファイル名に対して高精度な再処理（リファイン）を実行します。")
    args = parser.parse_args()

    print(f"🤖 Ollama ({OLLAMA_MODEL}モデル) を初期化します...")
    try:
        chat = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    except Exception as e:
        print(f"❌ Ollamaモデルの初期化に失敗しました。詳細: {e}"); return

    if args.refine:
        run_refine_process(chat, args.refine, document_state)
    else:
        try:
            all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.startswith("page-") and f.lower().endswith(".jpg")])
            run_basic_process(chat, all_files, document_state)
        except FileNotFoundError:
            print(f"❌ エラー: 入力ディレクトリ '{INPUT_DIR}' が見つかりません。")

    print("\n--- すべての処理が完了しました ---")

if __name__ == "__main__":
    main()
