import os
from pathlib import Path

# Ollama用のChatモデルをインポート
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

input_markdown_dir = 'outputs'
translated_markdown_dir = 'outputs_traslated'
translation_model = 'qwen3:32b'

def translate_markdown_file(filepath: Path, output_dir: Path, chain):
    """
    単一のMarkdownファイルを読み込み、翻訳して指定されたディレクトリに保存する。

    Args:
        filepath (Path): 翻訳するMarkdownファイルのパス。
        output_dir (Path): 翻訳済みファイルを保存するディレクトリのパス。
        chain: LangChainの翻訳チェーン。
    """
    print(f"🔄 処理中: {filepath}")

    try:
        # Markdownファイルの内容を読み込む
        content = filepath.read_text(encoding='utf-8')

        if not content.strip():
            print(f"❕ スキップ: {filepath} は空です。")
            return

        # LangChainを使って翻訳を実行
        # 長いテキストの場合、Ollamaからの応答に時間がかかることがあります
        translated_content = chain.invoke({"markdown_text": content})

        # 保存先のパスを決定
        relative_path = filepath.relative_to(Path('outputs'))
        output_path = output_dir / relative_path

        # 保存先のディレクトリが存在しない場合は作成
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 翻訳された内容をファイルに書き込む
        output_path.write_text(translated_content, encoding='utf-8')
        print(f"✅ 完了: {output_path}")

    except Exception as e:
        print(f"❌ エラー: {filepath} の処理中にエラーが発生しました - {e}")
        print("   Ollamaが起動しており、モデルがダウンロードされているか確認してください。")

def main():
    """
    メイン関数。ディレクトリをスキャンし、すべてのMarkdownファイルを翻訳する。
    """
    # --- LangChainのコンポーネントを設定 ---

    # 1. LLMモデルの初期化 (Ollama)
    # model="<モデル名>" の部分に、`ollama pull`でダウンロードしたモデル名を指定します。
    # 例: "llama3", "aya", "gemma2" など
    # temperature=0に設定することで、出力のランダム性を抑え、より安定した翻訳結果を得られます。
    try:
        model = ChatOllama(model=translation_model, temperature=0.1)
    except Exception as e:
        print(f"❌ エラー: Ollamaモデルの初期化に失敗しました - {e}")
        print("   Ollamaアプリケーションが正しく起動しているか確認してください。")
        return

    # 2. プロンプトテンプレートの作成
    # Ollamaで動かすローカルモデルでも同じプロンプトが有効です。
    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたはプロの翻訳家です。渡されたMarkdown形式のテキストを、英語から自然で高品質な日本語に翻訳してください。Markdownの構文（見出し、リスト、リンク、コードブロックなど）は、完全に維持してください。"),
        ("human", "{markdown_text}")
    ])

    # 3. 出力パーサーの初期化
    output_parser = StrOutputParser()

    # 4. チェーンの作成 (LCEL)
    translation_chain = prompt | model | output_parser

    # --- ファイル処理 ---
    source_dir = Path(input_markdown_dir)
    output_dir = Path(translated_markdown_dir) # 出力先ディレクトリ名を変更

    if not source_dir.is_dir():
        print(f"エラー: ソースディレクトリ '{source_dir}' が見つかりません。")
        return

    print("Ollamaを使用した翻訳処理を開始します...")
    print(f"使用モデル: {model.model}")
    print(f"入力元: {source_dir.resolve()}")
    print(f"出力先: {output_dir.resolve()}")

    # `outputs` ディレクトリ内のすべての.mdファイルを再帰的に検索
    markdown_files = list(source_dir.rglob('*.md'))

    if not markdown_files:
        print("翻訳対象のMarkdownファイルが見つかりませんでした。")
        return

    for filepath in markdown_files:
        translate_markdown_file(filepath, output_dir, translation_chain)

    print("\nすべてのファイルの翻訳が完了しました。")


if __name__ == '__main__':
    main()