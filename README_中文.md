# 免费 Receipt 自动入账工具（MacBook + 手机）

这个版本不需要 OpenAI API key，不需要付费。

## 功能

- 手机拍照/上传 receipt
- 本地 OCR 尝试识别 receipt 内容
- 页面中人工确认/修改每个 item
- 每个 item 一行写入同一个财年 Excel
- 自动保存 receipt 图片到 `~/Documents/Receipt_Records`
- Excel 中保存图片路径，方便 ATO 检查

## 第一次安装

```bash
cd Downloads/receipt_tool
pip3 install -r requirements.txt
```

如果 OCR 没有文字，需要安装 Tesseract：

```bash
brew install tesseract
```

如果你没有 Homebrew，先安装 Homebrew，或者也可以先不用 OCR，手动录入 item。

## 启动

```bash
cd Downloads/receipt_tool
streamlit run app.py
```

如果 `streamlit` 命令找不到：

```bash
python3 -m streamlit run app.py
```

## 手机使用

启动后 Terminal 会显示：

```text
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

手机和 MacBook 连接同一个 WiFi，然后用手机 Safari 打开 Network URL。

## 默认保存位置

```text
~/Documents/Receipt_Records/
  FY2025-2026/
    Bunnings_Warehouse/
      receipt-image.jpg
    Excel/
      FY2025-2026_Expense_Receipts.xlsx
```

## ATO 保存建议

把 `~/Documents/Receipt_Records` 同步到 iCloud Drive / Google Drive / OneDrive，避免电脑坏了丢失记录。
