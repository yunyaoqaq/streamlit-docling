import streamlit as st
import importlib
import logging
import platform
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Type
from enum import Enum

from docling_core.types.doc import ImageRefMode
from docling_core.utils.file import resolve_source_to_path
from pydantic import TypeAdapter

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    FormatToExtensions,
    InputFormat,
    OutputFormat,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    OcrEngine,
    OcrMacOptions,
    OcrOptions,
    PdfBackend,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")

_log = logging.getLogger(__name__)

# 定义文档导出函数（与原代码相同）
def export_documents(
    conv_results: list[ConversionResult],
    output_dir: Path,
    export_json: bool,
    export_html: bool,
    export_md: bool,
    export_txt: bool,
    export_doctags: bool,
    image_export_mode: ImageRefMode,
):
    success_count = 0
    failure_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            if export_json:
                fname = output_dir / f"{doc_filename}.json"
                _log.info(f"writing JSON output to {fname}")
                conv_res.document.save_as_json(filename=fname, image_mode=image_export_mode)

            if export_html:
                fname = output_dir / f"{doc_filename}.html"
                _log.info(f"writing HTML output to {fname}")
                conv_res.document.save_as_html(filename=fname, image_mode=image_export_mode)

            if export_txt:
                fname = output_dir / f"{doc_filename}.txt"
                _log.info(f"writing TXT output to {fname}")
                conv_res.document.save_as_markdown(
                    filename=fname, strict_text=True, image_mode=ImageRefMode.PLACEHOLDER
                )

            if export_md:
                fname = output_dir / f"{doc_filename}.md"
                _log.info(f"writing Markdown output to {fname}")
                conv_res.document.save_as_markdown(filename=fname, image_mode=image_export_mode)

            if export_doctags:
                fname = output_dir / f"{doc_filename}.doctags"
                _log.info(f"writing Doc Tags output to {fname}")
                conv_res.document.save_as_document_tokens(filename=fname)
        else:
            _log.warning(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
    return success_count, failure_count

# 分割语言列表的辅助函数
def _split_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or raw.strip() == "":
        return None
    return re.split(r"[;,]", raw.strip())

# Streamlit 主界面
def main():
    st.title("Docling 文档转换工具")
    st.write("上传文件并配置参数以将文档转换为指定格式。")

    # 支持的所有文件扩展名，基于 InputFormat 和 FormatToExtensions
    supported_extensions = []
    for fmt in InputFormat:
        extensions = FormatToExtensions.get(fmt, [])
        # 只添加与 InputFormat 对应的扩展名
        supported_extensions.extend(extensions)

    # 去重并转换为小写（避免大小写重复）
    supported_extensions = list(dict.fromkeys(supported_extensions))  # 去重
    supported_extensions = [ext.lower() for ext in supported_extensions]  # 统一为小写

    # 文件上传，支持所有 InputFormat 定义的格式
    uploaded_files = st.file_uploader(
        "上传文件",
        accept_multiple_files=True,
        type=supported_extensions,  # 使用精确的扩展名列表
        help=f"支持格式: {', '.join(supported_extensions)}"
    )

    # 输入格式选择（默认所有格式）
    from_formats = st.multiselect(
        "输入格式",
        options=[e.value for e in InputFormat],
        default=[e.value for e in InputFormat],
        help="选择允许的输入格式，默认支持所有格式。"
    )

    # 输出格式选择（支持多选）
    st.subheader("输出格式")
    export_json = st.checkbox("JSON", value=False)
    export_html = st.checkbox("HTML", value=False)
    export_md = st.checkbox("Markdown", value=True)  # 默认选中 Markdown
    export_txt = st.checkbox("TXT", value=False)
    export_doctags = st.checkbox("DocTags", value=False)

    # 图片导出模式
    image_export_mode = st.selectbox(
        "图片导出模式",
        options=[e.value for e in ImageRefMode],
        index=[e.value for e in ImageRefMode].index(ImageRefMode.EMBEDDED.value),
        help="选择图片在输出中的处理方式：占位符、嵌入或引用。"
    )

    # 输出目录
    output_dir = st.text_input("输出目录", value="./output", help="指定输出文件保存的目录。")

    # OCR 配置
    st.subheader("OCR 设置")
    do_ocr = st.checkbox("启用 OCR", value=True, help="处理位图内容时使用 OCR。")
    force_ocr = st.checkbox("强制 OCR", value=False, help="替换所有现有文本为 OCR 生成的文本。")
    ocr_engine = st.selectbox(
        "OCR 引擎",
        options=[e.value for e in OcrEngine],
        index=[e.value for e in OcrEngine].index(OcrEngine.EASYOCR.value),
        help="选择使用的 OCR 引擎。"
    )
    ocr_lang = st.text_input(
        "OCR 语言",
        value="",
        help="输入逗号或分号分隔的语言列表，例如 'en,zh'。"
    )

    # PDF 后端选择
    pdf_backend = st.selectbox(
        "PDF 后端",
        options=[e.value for e in PdfBackend],
        index=[e.value for e in PdfBackend].index(PdfBackend.DLPARSE_V2.value),
        help="选择 PDF 解析后端。"
    )

    # 表格模式
    table_mode = st.selectbox(
        "表格解析模式",
        options=[e.value for e in TableFormerMode],
        index=[e.value for e in TableFormerMode].index(TableFormerMode.ACCURATE.value),
        help="选择表格结构模型的模式。"
    )

    # 增强选项
    st.subheader("增强选项")
    enrich_code = st.checkbox("代码增强", value=False)
    enrich_formula = st.checkbox("公式增强", value=False)
    enrich_picture_classes = st.checkbox("图片分类", value=False)
    enrich_picture_description = st.checkbox("图片描述", value=False)

    # 其他高级选项
    st.subheader("高级设置")
    num_threads = st.slider("线程数", min_value=1, max_value=16, value=4)
    device = st.selectbox(
        "加速设备",
        options=[e.value for e in AcceleratorDevice],
        index=[e.value for e in AcceleratorDevice].index(AcceleratorDevice.AUTO.value)
    )
    document_timeout = st.number_input("文档处理超时（秒）", min_value=0.0, value=0.0, step=1.0, help="0 表示无超时限制。")
    verbose = st.select_slider("日志级别", options=["WARNING", "INFO", "DEBUG"], value="WARNING")

    # 调试选项
    st.subheader("调试选项")
    debug_visualize_cells = st.checkbox("可视化 PDF 单元格", value=False)
    debug_visualize_ocr = st.checkbox("可视化 OCR 单元格", value=False)
    debug_visualize_layout = st.checkbox("可视化布局", value=False)
    debug_visualize_tables = st.checkbox("可视化表格单元格", value=False)

    # 转换按钮
    if st.button("开始转换"):
        if not uploaded_files:
            st.error("请先上传文件！")
            return

        # 设置日志级别
        logging.basicConfig(level=getattr(logging, verbose))

        # 设置调试选项
        settings.debug.visualize_cells = debug_visualize_cells
        settings.debug.visualize_layout = debug_visualize_layout
        settings.debug.visualize_tables = debug_visualize_tables
        settings.debug.visualize_ocr = debug_visualize_ocr

        # 处理上传的文件
        with tempfile.TemporaryDirectory() as tempdir:
            input_doc_paths = []
            for uploaded_file in uploaded_files:
                temp_file_path = Path(tempdir) / uploaded_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.read())
                input_doc_paths.append(temp_file_path)

            # 转换格式为枚举类型
            from_formats_enum = [InputFormat(fmt) for fmt in from_formats]
            to_formats = []
            if export_json:
                to_formats.append(OutputFormat.JSON)
            if export_html:
                to_formats.append(OutputFormat.HTML)
            if export_md:
                to_formats.append(OutputFormat.MARKDOWN)
            if export_txt:
                to_formats.append(OutputFormat.TEXT)
            if export_doctags:
                to_formats.append(OutputFormat.DOCTAGS)

            if not to_formats:
                st.error("请选择至少一种输出格式！")
                return

            # 配置 OCR
            if ocr_engine == OcrEngine.EASYOCR.value:
                ocr_options = EasyOcrOptions(force_full_page_ocr=force_ocr)
            elif ocr_engine == OcrEngine.TESSERACT_CLI.value:
                ocr_options = TesseractCliOcrOptions(force_full_page_ocr=force_ocr)
            elif ocr_engine == OcrEngine.TESSERACT.value:
                ocr_options = TesseractOcrOptions(force_full_page_ocr=force_ocr)
            elif ocr_engine == OcrEngine.OCRMAC.value:
                ocr_options = OcrMacOptions(force_full_page_ocr=force_ocr)
            elif ocr_engine == OcrEngine.RAPIDOCR.value:
                ocr_options = RapidOcrOptions(force_full_page_ocr=force_ocr)
            else:
                st.error(f"未知的 OCR 引擎类型: {ocr_engine}")
                return

            ocr_lang_list = _split_list(ocr_lang)
            if ocr_lang_list:
                ocr_options.lang = ocr_lang_list

            # 配置管道选项
            accelerator_options = AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice(device))
            pipeline_options = PdfPipelineOptions(
                do_ocr=do_ocr,
                ocr_options=ocr_options,
                do_table_structure=True,
                do_code_enrichment=enrich_code,
                do_formula_enrichment=enrich_formula,
                do_picture_description=enrich_picture_description,
                do_picture_classification=enrich_picture_classes,
                document_timeout=document_timeout if document_timeout > 0 else None,
            )
            pipeline_options.table_structure_options.mode = TableFormerMode(table_mode)

            if ImageRefMode(image_export_mode) != ImageRefMode.PLACEHOLDER:
                pipeline_options.generate_page_images = True
                pipeline_options.generate_picture_images = True
                pipeline_options.images_scale = 2

            # 配置 PDF 后端
            if pdf_backend == PdfBackend.DLPARSE_V1.value:
                backend: Type[PdfDocumentBackend] = DoclingParseDocumentBackend
            elif pdf_backend == PdfBackend.DLPARSE_V2.value:
                backend = DoclingParseV2DocumentBackend
            elif pdf_backend == PdfBackend.PYPDFIUM2.value:
                backend = PyPdfiumDocumentBackend
            else:
                st.error(f"未知的 PDF 后端类型: {pdf_backend}")
                return

            pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options, backend=backend)
            format_options: Dict[InputFormat, FormatOption] = {
                InputFormat.PDF: pdf_format_option,
                InputFormat.IMAGE: pdf_format_option,
            }

            # 初始化转换器
            doc_converter = DocumentConverter(
                allowed_formats=from_formats_enum,
                format_options=format_options,
            )

            # 执行转换
            start_time = time.time()
            with st.spinner("正在转换文档..."):
                conv_results = list(doc_converter.convert_all(input_doc_paths, raises_on_error=False))
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                success_count, failure_count = export_documents(
                    conv_results,
                    output_dir=output_path,
                    export_json=export_json,
                    export_html=export_html,
                    export_md=export_md,
                    export_txt=export_txt,
                    export_doctags=export_doctags,
                    image_export_mode=ImageRefMode(image_export_mode),
                )
                end_time = time.time() - start_time

            # 显示结果
            st.success(f"转换完成！耗时: {end_time:.2f} 秒")
            st.write(f"成功处理 {success_count} 个文档，失败 {failure_count} 个文档。")
            st.write(f"结果已保存至: {output_dir}")

if __name__ == "__main__":
    main()