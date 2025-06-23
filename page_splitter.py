import csv
from pypdf import PdfReader, PdfWriter

def split_pdf_from_csv(input_pdf_path, csv_path, output_dir):
    """
    CSVファイルで定義された情報に基づいて、1つのPDFを複数のPDFに分割する。

    Args:
        input_pdf_path (str): 分割したい元のPDFファイルのパス
        csv_path (str): 分割情報を定義したCSVファイルのパス
    """
    try:
        # 元のPDFを読み込む
        reader = PdfReader(input_pdf_path)
        total_pages = len(reader.pages)
        print(f"'{input_pdf_path}' を読み込みました。(総ページ数: {total_pages})")

        # CSVファイルを読み込む
        with open(csv_path, mode='r', encoding='utf-8') as csvfile:
            # ヘッダー付きのCSVとして読み込む
            reader_csv = csv.DictReader(csvfile)
            
            for row in reader_csv:
                try:
                    output_filename = f"{output_dir}/{row['output_filename']}"
                    # ページ番号を整数に変換
                    start_page = int(row['start_page'])
                    end_page = int(row['end_page'])

                    # ページ範囲が妥当かチェック
                    if not (1 <= start_page <= end_page <= total_pages):
                        print(f"警告: '{output_filename}' のページ範囲({start_page}-{end_page})が不正です。スキップします。")
                        continue

                    # 新しいPDFを作成
                    writer = PdfWriter()

                    # ページ番号は0から始まるインデックスのため、-1 する
                    for i in range(start_page - 1, end_page):
                        writer.add_page(reader.pages[i])

                    # 新しいPDFファイルに書き出す
                    with open(output_filename, "wb") as output_file:
                        writer.write(output_file)
                    
                    print(f"✔️  '{output_filename}' を作成しました。 ({start_page}～{end_page}ページ)")

                except KeyError as e:
                    print(f"エラー: CSVファイルに '{e}' の列が見つかりません。")
                except ValueError:
                    print(f"エラー: CSVファイルの行 '{row}' のページ番号が不正です。スキップします。")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください: {e.filename}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

# --- 関数の実行例 ---
if __name__ == "__main__":
    # 設定項目
    input_pdf = "hott-ebook-15-ge428abf.pdf"    # 元のPDFファイル
    split_info_csv = "HoTT_PDF_sep.csv"           # 分割定義CSVファイル
    split_pdf_dir = "outputs_split"             # 分割後PDFの出力先ディレクトリ

    # スクリプトと、上記の2つのファイルが同じフォルダにあることを想定
    split_pdf_from_csv(input_pdf, split_info_csv, split_pdf_dir)