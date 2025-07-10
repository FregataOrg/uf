from .Log import Logger
from typing import Any, Optional
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import HumanMessage
from time import time

# --- 3. LLMClient クラス ---
class LLMClient:
    """LLM (Ollama or OpenRouter) との通信をカプセル化するクライアントクラス。"""
    def __init__(self, provider: str, model: str, temperature: float, logger: Logger,
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        self.logger = logger
        self.api_key = api_key
        self.base_url = base_url
        self.chat_model = self._initialize_model()

    def _initialize_model(self) -> Optional[Any]:
        self.logger.info(f"LLMプロバイダー '{self.provider}' のモデル '{self.model_name}' を初期化しています...")
        try:
            model_instance = None
            if self.provider == "openrouter":
                if not self.api_key:
                    self.logger.error("OpenRouter APIキーが設定されていません。")
                    return None
                model_instance = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=self.api_key,
                    base_url=self.base_url,
                    default_headers={
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "Image to Markdown Converter"
                    }
                )
            elif self.provider == "ollama":
                model_instance = ChatOllama(
                    model=self.model_name,
                    temperature=self.temperature,
                    base_url=self.base_url
                )
            else:
                self.logger.error(f"未対応のプロバイダーです: {self.provider}")
                return None

            self.logger.info("モデルとの疎通確認を実行中...")
            model_instance.invoke("hello")
            self.logger.success(f"{self.provider.capitalize()}モデルの初期化と疎通確認が完了しました。")
            return model_instance

        except Exception as e:
            self.logger.error(f"{self.provider.capitalize()}モデルの初期化に失敗しました。詳細を確認してください。\n詳細: {e}")
            return None

    def is_ready(self) -> bool: return self.chat_model is not None

    def invoke(self, prompt: str, base64_image: str) -> Optional[str]:
        if not self.is_ready(): self.logger.error("モデルが初期化されていないため、呼び出しをスキップします。"); return None
        self.logger.info("LLMにリクエストを送信しています..."); start_time = time()
        message = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}])
        try:
            response = self.chat_model.invoke([message])
            elapsed_time = time() - start_time
            self.logger.success(f"LLMから応答を受信しました。({elapsed_time:.2f}秒)"); return response.content
        except Exception as e: self.logger.error(f"モデルの呼び出し中にエラーが発生しました: {e}"); return None