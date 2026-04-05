"""Centralized settings for the Literature Smart Reader app."""

APP_TITLE = "文献智读"
APP_ICON = "📘"
APP_SUBTITLE = "面向课堂写作场景的文献学习辅助工具"

PAGE_HOME = "home"
PAGE_UPLOAD = "upload"
PAGE_RESULT = "result"
PAGE_ORDER = [PAGE_HOME, PAGE_UPLOAD, PAGE_RESULT]

ANALYSIS_STATUS_INITIAL = "initial"
ANALYSIS_STATUS_SELECTED = "selected"
ANALYSIS_STATUS_PARSING = "parsing"
ANALYSIS_STATUS_SUCCESS = "success"
ANALYSIS_STATUS_FAILED = "failed"

TEXT_PREVIEW_LENGTH = 2200
TEXT_PREVIEW_PARAGRAPHS = 3
MAX_KEYWORDS = 8
MIN_KEYWORDS = 3
MIN_TEXT_LENGTH = 80
SUMMARY_SENTENCE_MIN = 2
SUMMARY_SENTENCE_MAX = 4
SUMMARY_SOURCE_TEXT_LIMIT = 6000
TITLE_SCAN_TOP_RATIO = 0.35
TITLE_MAX_LINES = 2
LOW_INFORMATION_LINE_LENGTH = 40
LOW_INFORMATION_NON_TEXT_RATIO = 0.45

ENGLISH_STOPWORDS = {
    "about", "after", "also", "among", "analysis", "and", "are", "been", "between",
    "both", "can", "data", "each", "from", "have", "into", "more", "most", "paper",
    "result", "results", "show", "shows", "such", "than", "that", "their", "them",
    "there", "these", "this", "those", "using", "used", "with", "within", "without",
    "which", "while", "where", "when", "study", "research", "based", "method",
    "methods", "model", "approach", "through", "under", "over", "system"
}

CHINESE_STOP_PHRASES = {
    "研究", "本文", "方法", "结果", "问题", "数据", "分析", "进行", "提出",
    "基于", "通过", "发现", "影响", "采用", "表明", "认为", "以及", "相关",
    "可以", "使用", "论文", "课程", "文献", "摘要", "关键词", "作者"
}

METADATA_LINE_KEYWORDS = {
    "网络首发", "首发时间", "首发地址", "基金项目", "项目编号", "收稿日期", "修回日期",
    "录用日期", "作者简介", "通信作者", "文章编号", "中图分类号", "文献标识码", "doi",
    "issn", "cn", "作者单位", "online publication", "published online", "received", "accepted",
    "available online", "fund project", "corresponding author", "citation", "本文系", "本研究受"
}

METADATA_LINE_PATTERNS = [
    r"https?://\S+",
    r"\bwww\.[^\s]+",
    r"\bdoi\b\s*[:：]?\s*\S+",
    r"\bdoi\.org/\S+",
    r"(项目编号|基金编号|grant no\.?|project no\.?)\s*[:：]?\s*[A-Za-z0-9\-_.()/]+",
    r"(网络首发时间|online publication date|published online)\s*[:：]?\s*[\d]{4}[-/.年][\d]{1,2}[-/.月][\d]{1,2}",
    r"(收稿日期|received|accepted|修回日期|录用日期)\s*[:：]?\s*[\d]{4}[-/.年][\d]{1,2}[-/.月][\d]{1,2}",
    r"(本文系|本文为|本研究受|本论文受|本成果受)[^。；;\n]*(项目|基金|资助|课题)[^。；;\n]*",
    r"^\s*(?:第?\s*\d+\s*页|\d+\s*/\s*\d+)\s*$",
    r"^\s*(?:作者单位|作者简介|通信作者)\s*[:：]?\s*.+$",
    r"^\s*[*＊]?\s*\d+\s*[:：.]?\s*.+(?:大学|学院|研究院|研究所|实验室|department|university|college|institute).*$",
]

METADATA_FRAGMENT_PATTERNS = [
    r"https?://\S+",
    r"\bwww\.[^\s]+",
    r"(网络首发地址|available online(?: at)?|online publication url)\s*[:：]?\s*[^\n。；;]*",
    r"(网络首发时间|published online|online publication date)\s*[:：]?\s*[^\n。；;]*",
    r"(基金项目|fund project)\s*[:：]?\s*[^\n。；;]*",
    r"(项目编号|基金编号|grant no\.?|project no\.?)\s*[:：]?\s*[A-Za-z0-9\-_.()/]+",
    r"(收稿日期|received|accepted|修回日期|录用日期)\s*[:：]?\s*[^\n。；;]*",
    r"(作者单位|作者简介|通信作者|corresponding author)\s*[:：]?\s*[^\n。；;]*",
    r"(doi|issn|文章编号|中图分类号|文献标识码)\s*[:：]?\s*\S+",
    r"(本文系|本文为|本研究受|本论文受|本成果受)[^。\n；;]*(项目|基金|资助|课题)[^。\n；;]*",
    r"[*＊]?\s*\d+\s*[:：.]?\s*[^。\n；;]*(?:大学|学院|研究院|研究所|实验室|department|university|college|institute)[^。\n；;]*",
]

SUMMARY_NOISE_KEYWORDS = {
    "网络首发", "首发时间", "首发地址", "基金项目", "项目编号", "收稿日期", "修回日期",
    "录用日期", "作者简介", "通信作者", "doi", "issn", "文章编号", "中图分类号",
    "available online", "published online", "received", "accepted", "citation", "本文系", "本研究受"
}

STRUCTURED_LOW_CONFIDENCE_TEXT = "暂未识别到高置信度结果。"
STRUCTURED_LLM_TIMEOUT_SECONDS = 20
STRUCTURED_LLM_MAX_OUTPUT_TOKENS = 320
STRUCTURED_LLM_MIN_CANDIDATE_CHARS = 36
STRUCTURED_LLM_MIN_ABSTRACT_CHARS = 24
KEYWORD_LLM_MAX_OUTPUT_TOKENS = 160
COURSE_SUPPORT_LLM_MAX_OUTPUT_TOKENS = 900
COURSE_SUPPORT_LLM_MIN_SOURCE_CHARS = 120
