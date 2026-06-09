"""Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Bản hoàn chỉnh có fallback offline: nếu không cài crawl4ai hoặc không có mạng, script tạo JSON demo
đúng format để pipeline/test vẫn chạy được.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://www.24h.com.vn/giai-tri/hiep-ga-neu-khong-bi-bat-biet-dau-gio-toi-da-xanh-co-roi-c731a1687300.html",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://thanhnien.vn/dien-vien-hai-tran-huu-tin-lanh-7-nam-6-thang-tu-185230428134549434.htm",
    "https://congthuong.vn/chi-dan-an-tay-va-loat-nghe-si-dinh-den-ma-tuy-nghi-ve-trach-nhiem-cua-nguoi-noi-tieng-357972.html",
    "https://dantri.com.vn/phap-luat/nguoi-mau-an-tay-ru-ban-va-tro-ly-cung-su-dung-ma-tuy-20260406152426197.htm",
    "https://dantri.com.vn/phap-luat/ca-si-chi-dan-bi-dieu-tra-nghi-lien-quan-den-ma-tuy-20241110090028013.htm",
]

OFFLINE_ARTICLES = {
    ARTICLE_URLS[0]: ("Hiệp Gà: Nếu không bị bắt, biết đâu giờ tôi đã 'xanh cỏ’ rồi", "Bài viết là cuộc trò chuyện với diễn viên hài Hiệp Gà về quá khứ liên quan đến ma túy, quá trình bị bắt, chấp hành án và tái hòa nhập nghề diễn. Bài báo nhấn mạnh tác động của ma túy đến sự nghiệp, danh dự cá nhân và trách nhiệm của người nổi tiếng trước công chúng."),
    ARTICLE_URLS[1]: ("Sao Việt bị bắt, ngồi tù, mất danh tiếng vì chất cấm", "Bài viết điểm lại một số trường hợp người nổi tiếng trong showbiz Việt từng bị bắt, bị xét xử hoặc mất danh tiếng do liên quan đến chất cấm. Nội dung đặt các vụ việc vào chuỗi thời gian từ Hiệp Gà năm 2007 đến các trường hợp gần đây hơn như Hữu Tín, Chi Dân, An Tây và Nguyễn Công Trí."),
    ARTICLE_URLS[2]: ("Diễn viên hài Trần Hữu Tín lãnh 7 năm 6 tháng tù", "Bài báo tường thuật phiên tòa sơ thẩm vụ diễn viên hài Trần Hữu Tín. Tòa tuyên phạt Hữu Tín 7 năm 6 tháng tù về tội tổ chức sử dụng trái phép chất ma túy. Một bị cáo khác là Nguyễn Hoàng Phi bị tuyên mức án cao hơn do liên quan cả hành vi tàng trữ và tổ chức sử dụng trái phép chất ma túy."),
    ARTICLE_URLS[3]: ("Chi Dân, An Tây và loạt nghệ sĩ dính đến ma tuý: Nghĩ về trách nhiệm của người nổi tiếng", "Bài viết bàn về trách nhiệm của người nổi tiếng khi có thông tin liên quan đến ma túy. Theo bài báo, Công an TP.HCM đang tạm giữ Chi Dân và An Tây trong bối cảnh nhiều nghệ sĩ từng vướng tệ nạn xã hội. Bài báo đặt vấn đề rằng người nổi tiếng có ảnh hưởng đến công chúng nên cần ý thức rõ hơn về hành vi và hình ảnh cá nhân."),
    ARTICLE_URLS[4]: ("Người mẫu An Tây rủ bạn và trợ lý cùng sử dụng ma túy", "Bài báo nêu thông tin theo cáo trạng trong chuyên án VN10. Andrea Aybar / An Tây bị cáo buộc nhiều lần nhờ mua ma túy, chuẩn bị dụng cụ tại nơi ở và rủ bạn bè, trợ lý cùng sử dụng. Bài báo cho biết bị can bị truy tố về tội tổ chức sử dụng trái phép chất ma túy và tàng trữ trái phép chất ma túy."),
    ARTICLE_URLS[5]: ("Ca sĩ Chi Dân bị điều tra nghi liên quan đến ma túy", "Bài báo đưa tin ca sĩ Chi Dân bị lực lượng chức năng kiểm tra và phát hiện nghi vấn liên quan đến ma túy tại một địa chỉ ở TP.HCM. Thời điểm bài báo đăng, cơ quan chức năng đang mở rộng điều tra và chưa công bố kết quả xử lý cuối cùng."),
}


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _offline_article(url: str, i: int = 1) -> dict:
    title, summary = OFFLINE_ARTICLES.get(
        url,
        (f"Bài báo {i}: nghệ sĩ và vấn đề ma túy", "Tóm tắt tin tức phục vụ bài RAG, không sao chép toàn văn bài báo."),
    )
    content = (
        f"# {title}\n\n"
        "**Ghi chú:** Đây là dữ liệu fallback/paraphrase để pipeline chạy offline, "
        "không phải bản sao toàn văn bài báo.\n\n"
        "## Tóm tắt\n" + summary + "\n\n"
        "## Hướng dùng trong RAG\n"
        "Khi trả lời, cần nêu nguồn, phân biệt thông tin đã bị kết án với thông tin đang điều tra hoặc bị cáo buộc, "
        "và tránh suy đoán ngoài dữ kiện báo chí.\n"
    )
    return {"url": url, "title": title, "date_crawled": datetime.now().isoformat(), "content_markdown": content}


async def crawl_article(url: str) -> dict:
    """Crawl một bài báo; fallback offline khi thiếu crawl4ai/mạng."""
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return {
                "url": url,
                "title": getattr(result, "metadata", {}).get("title", "Unknown"),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": getattr(result, "markdown", ""),
            }
    except Exception:
        return _offline_article(url, 1)


async def crawl_all():
    setup_directory()
    for i, url in enumerate(ARTICLE_URLS, 1):
        article = await crawl_article(url)
        if not article.get("content_markdown"):
            article = _offline_article(url, i)
        (DATA_DIR / f"article_{i:02d}.json").write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ Saved article_{i:02d}.json")


if __name__ == "__main__":
    asyncio.run(crawl_all())
