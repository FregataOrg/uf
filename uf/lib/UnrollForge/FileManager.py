import os
import base64
import json
from .Log import Logger
from typing import List, Dict, Any, Optional
from PIL import Image
from io import BytesIO

# --- 2. FileManager クラス ---
class FileManager:
    """ファイルシステムの読み書き操作をすべて担当するクラス。"""
    def __init__(self, input_dir: str, output_dir: str, cache_dir: str, logger: Logger):
        self.input_dir = input_dir; self.output_dir = output_dir; self.cache_dir = cache_dir; self.logger = logger
    def get_cache_path(self, filename: str) -> str:
        return os.path.join(self.cache_dir, os.path.splitext(filename)[0] + ".json")
    def ensure_directories_exist(self):
        self.logger.info("必要なディレクトリの存在を確認・作成します...")
        try:
            for dir_path in [self.output_dir, self.cache_dir]:
                os.makedirs(dir_path, exist_ok=True)
                self.logger.success(f"ディレクトリ '{dir_path}' の準備ができました。")
        except OSError as e: self.logger.error(f"ディレクトリ作成中にエラーが発生しました: {e}"); raise
    def get_image_files(self) -> List[str]:
        self.logger.info(f"入力ディレクトリ '{self.input_dir}' から画像ファイルを検索します...")
        try:
            files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            self.logger.success(f"{len(files)} 件の画像ファイルが見つかりました。")
            return files
        except FileNotFoundError: self.logger.error(f"入力ディレクトリ '{self.input_dir}' が見つかりません。"); return []
        except Exception as e: self.logger.error(f"ファイルリストの取得中に予期せぬエラーが発生しました: {e}"); return []
    def read_image_as_base64(self, filename: str) -> Optional[str]:
        image_path = os.path.join(self.input_dir, filename)
        try:
            with Image.open(image_path) as img:
                buffered = BytesIO(); img.convert('RGB').save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except FileNotFoundError: self.logger.error(f"画像ファイル '{image_path}' が見つかりません。"); return None
        except Exception as e: self.logger.error(f"画像処理中にエラーが発生しました ({filename}): {e}"); return None
    def write_markdown(self, filename: str, content: str):
        base_name = os.path.splitext(filename)[0]; output_path = os.path.join(self.output_dir, f"{base_name}.md")
        try:
            with open(output_path, 'w', encoding='utf-8') as f: f.write(content)
            self.logger.success(f"Markdownファイル '{output_path}' を保存しました。")
        except IOError as e: self.logger.error(f"Markdownファイルの書き込みに失敗しました ({output_path}): {e}")
    def read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(file_path): return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
        except (json.JSONDecodeError, IOError) as e: self.logger.error(f"JSONファイルの読み込みに失敗しました ({file_path}): {e}"); return None
    def write_json(self, file_path: str, data: Dict[str, Any]):
        try:
            with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e: self.logger.error(f"JSONファイルの書き込みに失敗しました ({file_path}): {e}")