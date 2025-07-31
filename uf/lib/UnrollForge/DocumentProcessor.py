from typing import Dict, Any, Tuple, List, Optional
from re import findall, MULTILINE
from .FileManager import FileManager
from .Log import Logger
from .LLMClient import LLMClient
from os import path

SYSTEM_PROMPT = """
You are an expert digital archivist specializing in mathematical and scientific texts. Your task is to perform high-fidelity Optical Character Recognition (OCR) and document layout analysis, converting physical pages into perfectly structured Markdown documents with accurate LaTeX formatting.
"""
BASIC_PROMPT = """{}\n{}\n
Your task is to transcribe the provided page into a clean Markdown document with perfect LaTeX formatting.\n
Here is an example of the quality and format require:\n
--- EXAMPLE START ---\n
{}\n
--- EXAMPLE END ---\n
"""

PERFECT_EXAMPLE_MARKDOWN = """
### Example Inline Mathematical Formula

This is an example of an inline mathematical formula: $E=mc^2$.

### Exapmle Block Mathematical Formula

This is an example of a block mathematical formula:
```math
\int_0^1 x^2 \, dx = \frac{1}{3}
```

### Example of a Table Content

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1 Col 1 | Row 1 Col 2 | Row 1 Col 3 |
| Row 2 Col 1 | Row 2 Col 2 | Row 2 Col 3 |

### Example of a Figure Caption
![Example Figure](placeholder.png)

**This is a placeholder for an image caption. The actual image will be processed and inserted later.**
"""

# --- 4. DocumentState クラス ---
class DocumentState:
    """ドキュメント全体の処理状態と構造的文脈（コンテキスト）を管理するクラス。"""
    def __init__(self, state_file_path: str, file_manager: FileManager, logger: Logger):
        self.state_file_path = state_file_path; self.file_manager = file_manager; self.logger = logger
        self._state: Dict[str, Any] = self._get_default_state(); self.load()
    def _get_default_state(self) -> Dict[str, Any]: return {"last_processed_page": None, "heading_context_stack": []}
    def load(self):
        self.logger.info(f"ドキュメント状態を '{self.state_file_path}' から読み込みます。")
        loaded_state = self.file_manager.read_json(self.state_file_path)
        if loaded_state and "last_processed_page" in loaded_state and "heading_context_stack" in loaded_state:
            self._state = loaded_state; self.logger.success("ドキュメント状態の読み込みに成功しました。")
        else:
            self.logger.info("既存の状態ファイルが見つからないか形式が不正なため、新しいドキュメントとして開始します。"); self._state = self._get_default_state()
    def save(self):
        self.logger.info(f"現在のドキュメント状態を '{self.state_file_path}' に保存します。")
        self.file_manager.write_json(self.state_file_path, self._state)
    def update_heading_stack(self, markdown_content: str):
        stack = self._state.get("heading_context_stack", []); headings = findall(r'^(#+)\s+(.*)', markdown_content, MULTILINE)
        if not headings: self.logger.info("このページには新しい見出しがありませんでした。コンテキストは変更されません。"); return
        for heading_marks, title in headings:
            level = len(heading_marks); new_heading = {"level": level, "title": title.strip()}
            while stack and stack[-1]['level'] >= level: stack.pop()
            stack.append(new_heading)
        self._state["heading_context_stack"] = stack; self.logger.success(f"見出しコンテキストを更新しました。")
    def get_context_prompt(self) -> Tuple[str, int]:
        stack = self._state.get("heading_context_stack", [])
        if not stack: context_str = "No current context (This is likely the beginning of the document)."; depth = 0
        else: path_str = " > ".join([f"{'#' * h['level']} {h['title']}" for h in stack]); depth = stack[-1]['level']; context_str = f"Current Path: {path_str}\nCurrent Heading Depth: {depth}"
        return f"\n--- DOCUMENT CONTEXT ---\n...\n{context_str}\n--- END DOCUMENT CONTEXT ---\n", depth
    def get_last_processed_page(self) -> Optional[str]: return self._state.get("last_processed_page")
    def set_last_processed_page(self, filename: str): self.logger.info(f"処理済みページを '{filename}' に更新しました。"); self._state["last_processed_page"] = filename

# --- 5. DocumentProcessor クラス (オーケストレーター) ---
class DocumentProcessor:
    """各コンポーネントを協調させ、ドキュメント処理のワークフロー全体を制御する。"""
    
    PERFECT_EXAMPLE_MARKDOWN = """
...as shown in the equation:

$$ \sum_{i=0}^{n} i = \frac{n(n+1)}{2} $$

This is followed by more text.
"""

    def __init__(self, file_manager: FileManager, logger: Logger, llm_client: LLMClient, doc_state: DocumentState):
        self.file_manager = file_manager
        self.logger = logger
        self.llm_client = llm_client
        self.doc_state = doc_state

    def _analyze_page_structure(self, filename: str, context_prompt: str) -> Optional[str]:
        """キャッシュを利用しつつ、ページの構造をLLMで解析する (リファインモード用)。"""
        cache_path = self.file_manager.get_cache_path(filename)
        cached_data = self.file_manager.read_json(cache_path)
        if cached_data and "analysis_text" in cached_data:
            self.logger.success(f"キャッシュ利用: {path.basename(cache_path)}")
            return cached_data["analysis_text"]

        self.logger.info(f"ページ構造を解析中: {filename}")
        b64_image = self.file_manager.read_image_as_base64(filename)
        if not b64_image: return None

        analysis_prompt = f"""
        {SYSTEM_PROMPT}
        {context_prompt}
        Analyze the layout of this page by following these steps: 
        1. **Overall Layout**: First, identify the overall layout. Is it single-column, two-colmun, or something more complex?
        2. **Component Identification**: Second, locate and identify all distinct components: headers, footers, main text body, figures, tables, and most importantly, mathematical formula blocks.
        3. **Challenge Assessment**: Third, note any potential challenges for transcription, such as small font sizes, comples nested formulas, or unusual text flow.
        Provide your analysis as a concise JSON object, forcusing only on the structural facts.
"""
        analysis_text = self.llm_client.invoke(analysis_prompt, b64_image)

        if analysis_text:
            self.file_manager.write_json(cache_path, {"analysis_text": analysis_text})
        return analysis_text

    def run_basic(self):
        """未処理の全ファイルを対象に、基本的なOCR処理を順次実行する。"""
        self.logger.start_section("🚀 基本処理モード 開始")
        all_files = self.file_manager.get_image_files()
        last_processed = self.doc_state.get_last_processed_page()
        start_index = all_files.index(last_processed) + 1 if last_processed in all_files else 0
        if start_index > 0: self.logger.info(f"前回の処理 '{last_processed}' の次から再開します。")

        files_to_run = all_files[start_index:]
        if not files_to_run: self.logger.success("すべてのファイルは既に処理済みです。"); return

        for filename in files_to_run:
            self.logger.start_section(f"ターゲット: {filename}")
            context_prompt, _ = self.doc_state.get_context_prompt()
            basic_prompt = BASIC_PROMPT.format(SYSTEM_PROMPT, context_prompt, self.PERFECT_EXAMPLE_MARKDOWN)

            b64_image = self.file_manager.read_image_as_base64(filename)
            if not b64_image: continue

            markdown_content = self.llm_client.invoke(basic_prompt, b64_image)
            if markdown_content:
                self.file_manager.write_markdown(filename, markdown_content)
                self.doc_state.update_heading_stack(markdown_content)
                self.doc_state.set_last_processed_page(filename)
                self.doc_state.save()
            else:
                self.logger.error(f"処理失敗: {filename}。処理を中断します。"); break

    def run_refine(self, refine_files: List[str]):
        """指定されたファイルリストに対して、高精度な再処理を実行する。"""
        self.logger.start_section("✨ 再処理（リファイン）モード 開始")
        self.logger.warn("このモードは文書全体の状態（最終処理ページなど）を更新しません。")

        for filename in refine_files:
            self.logger.start_section(f"ターゲット: {filename}")
            context_prompt, _ = self.doc_state.get_context_prompt()

            analysis_text = self._analyze_page_structure(filename, context_prompt)
            if not analysis_text: self.logger.error(f"解析失敗: {filename}"); continue

            b64_image = self.file_manager.read_image_as_base64(filename)
            if not b64_image: continue

            refined_prompt = f"""{SYSTEM_PROMPT}
{context_prompt}
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

            markdown_content = self.llm_client.invoke(refined_prompt, b64_image)
            if markdown_content:
                self.file_manager.write_markdown(filename, markdown_content)
