from lib.UnrollForge.Log import Logger
from lib.UnrollForge.FileManager import FileManager
from lib.UnrollForge.LLMClient import LLMClient
from lib.UnrollForge.DocumentProcessor import DocumentProcessor, DocumentState
from os import path
import json
import argparse

# argparse
def parse_args():
    parser = argparse.ArgumentParser(description="画像化された本を構造化Markdownに変換します。")
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help="設定ファイルのパス (例: config.json)"
    )
    parser.add_argument('--refine', nargs='+', metavar='FILENAME', help="指定ファイルに高精度な再処理を実行。 例: --refine file1.md file2.md")
    return parser.parse_args()

# --- 6. メイン実行ブロック ---
def main():
    """
    設定ファイルを読み込み、アプリケーションを初期化し、コマンドライン引数に基づいて処理を実行する。
    """
    # --- コマンドライン引数の解析 ---
    args = parse_args()

    # --- 設定ファイルの読み込み ---
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ エラー: 設定ファイル '{args.config}' が見つかりません。")
        return
    except json.JSONDecodeError:
        print(f"❌ エラー: 設定ファイル '{args.config}' のJSON形式が正しくありません。")
        return

    # --- 依存関係の構築 (Dependency Injection) ---
    logger = Logger()
    logger.start_section("アプリケーション初期化")

    # 設定ファイルから設定値を取得
    dirs = config.get("directories", {})
    file_manager = FileManager(
        input_dir=dirs.get("input", "input"),
        output_dir=dirs.get("output", "output"),
        cache_dir=dirs.get("cache", "cache"),
        logger=logger
    )
    file_manager.ensure_directories_exist()

    # アクティブなプロバイダーの設定を取得
    active_provider_name = config.get("active_provider", "ollama")
    provider_settings = config.get("providers", {}).get(active_provider_name, {})

    logger.info(f"使用プロバイダー: {active_provider_name.upper()}, モデル: {provider_settings.get('model')}")

    llm_client = LLMClient(
        provider=active_provider_name,
        model=provider_settings.get("model"),
        temperature=config.get("temperature", 0.1),
        logger=logger,
        api_key=provider_settings.get("api_key"),
        base_url=provider_settings.get("base_url")
    )

    if not llm_client.is_ready():
        logger.error("LLMクライアントの準備ができなかったため、処理を終了します。")
        return

    state_file_path = path.join(dirs.get("output", "output"), config.get("state_file_name", "state.json"))
    doc_state = DocumentState(
        state_file_path=state_file_path,
        file_manager=file_manager,
        logger=logger
    )

    processor = DocumentProcessor(
        file_manager=file_manager,
        logger=logger,
        llm_client=llm_client,
        doc_state=doc_state
    )

    # --- 処理の実行 ---
    # --- 処理の実行 ---
    if args.refine:
        processor.run_refine(args.refine)
    else:
        processor.run_basic()

    logger.start_section("🎉 すべての処理が完了しました 🎉")

# argparse の部分は完全に削除し、main 関数を直接呼び出すように変更します。
if __name__ == "__main__":
    main()
