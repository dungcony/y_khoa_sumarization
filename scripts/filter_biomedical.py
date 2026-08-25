#!/usr/bin/env python3
"""Keep human biomedical summarization samples and audit every exclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SPLITS = ("train", "validation", "test")
WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def term_set(*values: str) -> frozenset[str]:
    return frozenset(normalize_for_matching(value).strip() for value in values)


def normalize_for_matching(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return f" {WORD_RE.sub(' ', text).strip()} "


HUMAN_TERMS = term_set(
    "người", "con người", "trẻ", "trẻ em", "em bé", "bé trai", "bé gái",
    "nam giới", "nữ giới", "phụ nữ", "đàn ông", "bệnh nhân", "nạn nhân",
    "sản phụ", "thai phụ", "người bệnh", "người cao tuổi", "người già",
    "bạn", "học sinh", "sinh viên", "phụ huynh", "gia đình", "cháu",
    "em nhỏ", "bé", "cụ", "ông", "chị", "nữ chủ", "ngư dân",
    "bà cụ", "bà mẹ", "người bà", "bà lão", "bệnh nhi",
)

DISEASE_TERMS = term_set(
    "bệnh lý", "hội chứng", "ung thư", "khối u", "u ác tính",
    "viêm", "nhiễm trùng", "nhiễm khuẩn", "nhiễm virus", "nhiễm vi rút",
    "dị ứng", "rối loạn", "ngộ độc", "đột quỵ", "tiểu đường",
    "đái tháo đường", "tăng huyết áp", "hạ huyết áp", "hiv", "aids",
    "covid", "sốt rét", "rubella", "sởi", "cúm", "lao phổi", "bạch cầu",
    "suy tim", "suy thận", "suy gan", "suy hô hấp", "tâm thần",
    "trầm cảm", "tự kỷ", "alzheimer", "parkinson", "động kinh",
    "loãng xương", "thoát vị", "sỏi thận", "mụn cóc", "mụn cơm",
    "nấm da", "herpes", "giang mai", "chlamydia", "mãn tính", "mạn tính",
    "bệnh truyền nhiễm", "dịch bệnh", "bệnh dịch", "bệnh dại",
    "phình động mạch", "sán dây", "ký sinh trùng", "suy dinh dưỡng",
    "thiếu dinh dưỡng", "nhiễm độc", "sốc phản vệ", "bong gân", "gãy xương",
    "đau nửa đầu", "viêm gan", "viêm phổi", "viêm họng", "viêm da",
    "viêm xoang", "viêm khớp", "viêm ruột", "viêm loét", "loét dạ dày",
    "nhiễm hiv", "hpv", "sùi mào gà", "sán lợn", "sán chó", "ho gà",
    "cảm lạnh", "cảm cúm", "cao huyết áp", "huyết áp cao", "huyết áp thấp",
    "mỡ máu", "thiếu máu", "bệnh tim", "bệnh phổi", "bệnh gan",
    "bệnh thận", "bệnh da", "bệnh răng miệng", "bệnh đường ruột",
    "mụn trứng cá", "nghiện", "cơn hoảng loạn", "cơn hoảng sợ",
    "bóng đè", "hoại thư", "mất thính lực",
)

HUMAN_ZOONOTIC_DISEASE_TERMS = term_set(
    "sán lợn", "sán chó", "sán dây", "bệnh dại", "cúm", "ngộ độc",
    "ký sinh trùng",
)

TREATMENT_TERMS = term_set(
    "chẩn đoán", "điều trị", "chữa trị", "chữa bệnh", "phẫu thuật",
    "liệu pháp", "uống thuốc", "dùng thuốc", "thuốc kê", "thuốc kháng",
    "kháng sinh", "vắc xin", "vắc-xin", "vaccine", "tiêm chủng",
    "xét nghiệm", "ghép tạng", "hiến tạng", "truyền máu", "hồi sức cấp cứu",
    "hồi sức tích cực", "khoa hồi sức",
    "cấy ghép", "kê đơn", "kê toa", "lọc máu", "hóa trị", "xạ trị",
    "chăm sóc y tế", "tư vấn y tế", "phục hồi chức năng",
    "sử dụng thuốc", "bôi thuốc", "thoa thuốc", "thuốc giảm đau",
    "thuốc nhỏ mắt", "thuốc mỡ", "thuốc đặc trị", "thuốc nhuận tràng",
    "thuốc tránh thai", "thuốc lợi tiểu", "thuốc ức chế", "xịt mũi",
    "thuốc sát trùng", "bút tiêm", "epipen",
    "thuốc bán sẵn", "thuốc theo chỉ định", "sát trùng", "hiến máu",
    "viên ngậm trị ho", "thuốc ho", "sáp nha khoa",
    "tiêm ngừa", "tiêm phòng", "văcxin", "vắcxin", "khám da liễu",
    "khám nha khoa", "nha sĩ", "sơ cứu vết thương", "chăm sóc vết thương",
)

CLINICAL_TERMS = term_set(
    "triệu chứng", "cấp cứu", "nhập viện", "đi khám", "khám bệnh",
    "gặp bác sĩ", "hỏi bác sĩ", "trao đổi với bác sĩ",
    "tham khảo ý kiến bác sĩ", "sản phụ", "thai phụ",
    "tiên lượng", "theo dõi sức khỏe", "theo dõi sức khoẻ",
)

HEALTH_SYSTEM_TERMS = term_set(
    "bác sĩ", "bệnh viện", "bộ y tế", "sở y tế", "ngành y tế",
    "cơ sở y tế", "nhân viên y tế", "bảo hiểm y tế", "sức khỏe",
    "sức khoẻ", "y khoa", "y học", "y sinh", "dược", "nhà thuốc",
    "phòng khám", "trung tâm y tế", "trạm y tế", "chuyên gia y tế",
    "dịch tễ", "sức khỏe cộng đồng", "sức khoẻ cộng đồng",
)

ANATOMY_TERMS = term_set(
    "cơ thể người", "não người", "hệ thần kinh", "hệ miễn dịch",
    "nhiễm sắc thể", "tế bào", "adn", "dna", "gen", "di truyền", "tim",
    "phổi", "gan", "thận", "dạ dày", "đường ruột", "ruột", "tuyến ức",
    "tuyến giáp", "niệu quản", "bàng quang", "tử cung", "buồng trứng",
    "âm đạo", "dương vật", "tinh hoàn", "tinh trùng", "kinh nguyệt",
    "mang thai", "thai nhi", "sơ sinh", "xương", "khớp", "cột sống",
    "máu", "huyết", "hormone", "hoóc môn", "protein", "enzyme",
    "não", "mũi", "họng", "răng", "miệng", "tai", "mắt", "da",
    "tuyến vú", "bạch huyết", "động mạch", "tĩnh mạch", "nội tạng", "lưng",
    "hậu môn", "trực tràng", "cổ tử cung", "thai kỳ", "tiền sản",
    "vùng kín",
)

SYMPTOM_TERMS = term_set(
    "đau", "sốt", "ho", "nôn", "buồn nôn", "chóng mặt", "khó thở",
    "chảy máu", "bỏng", "gãy", "vết thương", "chấn thương", "sưng",
    "phù", "ngứa", "phát ban", "mụn", "mất ngủ", "mệt mỏi", "co giật",
    "tê", "tử vong", "chết não", "tiêu chảy", "táo bón", "khò khè",
    "nghẹt mũi", "đau bụng", "đau đầu", "đau ngực", "đau lưng",
    "đau họng", "ngất", "khó nuốt", "bầm tím", "vết bầm", "chảy nước mũi",
    "vết phồng", "phồng rộp", "mụn nước", "bong vảy",
)

PREVENTION_TERMS = term_set(
    "dinh dưỡng", "giảm cân", "béo phì", "chế độ ăn",
    "ăn uống lành mạnh", "tập thể dục", "vận động", "giấc ngủ",
    "vệ sinh cá nhân", "sinh sản", "tránh thai", "tình dục an toàn",
    "cai thuốc", "bỏ hút thuốc", "rượu bia", "sơ cứu", "phòng bệnh",
    "ngăn ngừa bệnh", "phòng ngừa", "chăm sóc sức khỏe",
    "chăm sóc sức khoẻ", "sức khỏe tâm thần", "sức khoẻ tâm thần",
    "rửa tay", "khử trùng", "vệ sinh răng miệng", "chăm sóc răng miệng",
    "đánh răng", "chải răng", "chỉ nha khoa",
)

RESEARCH_TERMS = term_set(
    "nghiên cứu", "nhà khoa học", "thử nghiệm lâm sàng", "công nghệ y tế",
    "công nghệ sinh học", "sinh học phân tử", "kỹ thuật y sinh",
)

ANIMAL_TERMS = term_set(
    "động vật", "vật chủ", "loài", "chó", "mèo", "chim", "côn trùng",
    "bò sát", "thú cưng", "thú nuôi", "giống chó", "giống mèo", "con vật",
    "con bò", "trâu", "con ngựa", "khỉ", "gấu", "rắn", "con trăn", "hổ",
    "voi", "rùa", "ong đốt", "ong bắp cày", "nhện", "loài bướm", "con cua",
    "con tôm", "lợn", "heo", "con gà", "gà mái", "gia cầm", "vịt", "dơi",
    "sâu bướm", "thỏ", "con chuột", "chuột lang", "hamster", "vẹt",
    "con cá", "loài cá", "cá voi", "cá mập", "cá sấu", "bể cá", "nuôi cá",
    "cá chết", "cá lồng", "cá nuôi", "mẫu cá", "con cóc", "thịt cóc",
    "gan cóc", "ăn cóc", "báo tuyết",
    "tinh tinh", "linh trưởng",
)

HUMAN_EXPOSURE_ANIMAL_TERMS = term_set(
    "ong đốt", "ong bắp cày", "nhện", "rắn", "mèo cào", "bị chó cắn",
    "chó cắn", "hổ cắn",
)

CORE_RESEARCH_ANATOMY_TERMS = term_set(
    "cơ thể người", "não người", "hệ thần kinh", "hệ miễn dịch",
    "nhiễm sắc thể", "tế bào", "adn", "dna", "gen", "di truyền",
    "não", "tim", "phổi", "gan", "dạ dày", "đường ruột", "ruột",
    "tuyến giáp", "tuyến ức", "niệu quản", "bàng quang", "tử cung",
    "buồng trứng", "tinh trùng", "thai nhi", "sơ sinh", "máu",
    "hormone", "hoóc môn", "protein", "enzyme", "bạch huyết",
    "động mạch", "tĩnh mạch",
)

VETERINARY_TERMS = term_set(
    "bác sĩ thú y", "bác sỹ thú y", "bác sĩ thú ý", "phòng khám thú y", "đi khám thú y",
    "thú y", "chăm sóc chó", "chăm sóc mèo", "huấn luyện chó",
    "huấn luyện mèo", "cho chó", "cho mèo", "đưa chó", "đưa mèo",
    "mang chó", "mang mèo", "nuôi chó", "nuôi mèo",
    "chó bị", "mèo bị", "bệnh ở chó", "bệnh ở mèo", "sức khỏe chó",
    "sức khoẻ chó", "sức khỏe mèo", "sức khoẻ mèo", "chó của bạn",
    "sức khỏe của chó", "sức khoẻ của chó", "sức khỏe của mèo",
    "sức khoẻ của mèo",
    "mèo của bạn", "điều trị cho chó", "điều trị cho mèo", "thuốc cho chó",
    "thuốc cho mèo", "tiêm cho chó", "tiêm cho mèo", "chăm sóc thú cưng",
    "cá bị", "cá mắc", "bệnh ở cá", "điều trị cho cá", "thuốc cho cá",
    "cho cá", "cá betta", "cá vàng", "vịt bị", "vịt mắc", "bệnh ở vịt",
    "điều trị cho vịt", "thuốc cho vịt", "cho vịt", "gà bị", "gà mắc",
    "bệnh ở gà", "điều trị cho gà", "thuốc cho gà", "cho gà",
    "bệnh ở ngựa", "điều trị cho ngựa", "thuốc cho ngựa", "cho ngựa",
    "dịch tả lợn", "dịch tả heo", "vắc xin cho lợn", "vaccine cho lợn",
    "bệnh ở cá", "rối loạn bong bóng khi bơi ở cá", "loại cá nào",
)

TAXONOMY_TERMS = term_set(
    "danh pháp", "tên khoa học", "thuộc họ", "thuộc bộ", "thuộc chi",
    "một loài", "phân loài", "tuyệt chủng", "sinh cảnh", "phân bố",
    "hoang dã", "bảo tồn động vật", "hệ sinh thái",
)

PLANT_TERMS = term_set(
    "trồng cây", "chăm sóc cây", "tỉa cây", "tưới cây", "bón cây",
    "sâu bệnh cho cây", "nhận dạng cây", "loài thực vật", "cây cảnh",
    "hoa mẫu đơn", "vườn cây", "làm vườn", "phun thuốc cho cây",
    "thuốc diệt côn trùng", "thuốc trừ sâu", "thuốc bảo vệ thực vật",
    "giống cây trồng", "cây trồng", "hoa cảnh",
)

INCIDENT_TERMS = term_set(
    "tai nạn", "tấn công", "điện giật", "đuối nước", "sét đánh", "bị chém",
    "bị đâm", "bị đánh", "nhảy lầu", "rơi xuống", "thi thể", "công an",
    "khởi tố", "bắt giữ", "trộm", "cướp", "án mạng", "tử nạn",
    "treo cổ", "tự tử", "tự sát", "rơi lầu", "nhảy cầu", "nhảy xuống",
    "vụ cháy", "vụ hỏa hoạn", "vụ hoả hoạn", "vụ nổ", "nổ pháo",
    "va chạm", "bị tông", "tông trúng", "tông quay", "tông vào", "tông",
    "bị bắn", "viên đạn", "đạn xuyên",
    "nhảy từ", "nhảy qua cửa sổ", "chập điện", "xẹt lửa", "bốc cháy",
    "đè vào người", "nổ bình ga", "nổ bình gas",
    "phát nổ", "cú đối đầu", "vết dao đâm",
)

LEGAL_OR_CRIME_TERMS = term_set(
    "công an", "cảnh sát", "khởi tố", "bắt giữ", "tạm giữ", "truy tố",
    "xét xử", "toà án", "tòa án", "ra toà", "ra tòa",
    "án mạng", "giết người", "cưỡng hiếp", "trộm", "cướp", "bắt cóc",
    "hành hung", "kỷ luật", "sai phạm", "tham nhũng", "lừa đảo",
    "băng nhóm", "gây mất trật tự", "bỏ trốn", "dùng súng",
    "cưỡng bức", "xâm hại", "chém", "đập phá", "xông vào", "bị cáo",
    "án tù", "tù treo", "đề nghị truy tố", "nâng khống", "đem bán",
    "lăng mạ", "bạo hành", "đơn khiếu nại", "cơ quan điều tra",
    "cò máu", "cò bệnh viện", "cò khám bệnh", "tố bác sĩ",
    "bức xúc phản ánh",
    "sờ soạng", "bị can", "nghi phạm", "nghi can", "giết", "sát hại",
    "bẫy mìn", "dùng gậy", "bị bạn cắn", "bị bạn đánh", "cáo buộc",
    "bị phạt", "hạ sát", "tưới xăng", "đốt do ghen", "gây án",
    "đường dây", "luật sư", "thẩm vấn", "lừa", "dùng dao",
    "làm giả", "tuyên phạt", "bê bối",
)

ADMINISTRATIVE_TERMS = term_set(
    "bổ nhiệm", "miễn nhiệm", "nguyên giám đốc bệnh viện",
    "trưởng ban", "quyết định bổ nhiệm", "thu phí", "viện phí", "ngân sách",
    "đấu thầu", "nghị định", "thanh tra", "kiểm điểm",
    "vi phạm quy định", "quy định pháp luật", "ban hành quy định",
    "khai trương", "khánh thành", "bệnh viện có quy mô", "giường bệnh", "thành tựu",
    "trao giải", "giải thưởng", "được phân công", "nhận nhiệm vụ",
    "tự chủ bệnh viện", "tự chủ toàn diện", "thuộc bộ y tế",
    "kết nối dữ liệu", "đào tạo bác sĩ", "đào tạo chuyên khoa",
    "khai giảng lớp bác sĩ", "thầy thuốc trẻ tiêu biểu", "bác sĩ trẻ tình nguyện",
    "gửi xe", "bồi thường", "giá dịch vụ", "chi trả", "thanh toán",
    "nguồn vốn", "xuống cấp", "mở rộng bệnh viện",
    "điều chuyển", "giữ chức",
    "bất cập", "bất tiện", "viện trợ",
)

STRONG_ADMINISTRATIVE_TERMS = term_set(
    "bổ nhiệm", "miễn nhiệm", "quyết định bổ nhiệm", "được phân công",
    "nhận nhiệm vụ", "giữ chức", "thu phí", "viện phí", "ngân sách",
    "đấu thầu", "nghị định", "giá dịch vụ", "chi trả", "thanh toán",
    "kết nối dữ liệu", "khai trương", "khánh thành", "bệnh viện có quy mô",
    "giường bệnh", "nguồn vốn", "xuống cấp", "gửi xe", "bồi thường",
    "cò máu", "cò bệnh viện", "cò khám bệnh",
    "ra mắt", "bảo hiểm y tế", "bhxh", "thu hồi giấy phép", "tước giấy phép",
    "hãng taxi", "xây dựng bệnh viện", "cao tốc", "thay đổi thiết kế",
    "ngành lao động", "chính sách", "đại biểu quốc hội", "đuổi việc",
    "xử phạt",
    "giấy chứng nhận", "lãng phí", "khoán chi", "kinh phí",
    "công nhận kết quả xét nghiệm", "tạm đình chỉ", "phó thủ tướng",
    "đình chỉ công tác", "phá sản", "tài chính",
)

DECEPTIVE_HEALTH_TERMS = term_set(
    "giả vờ", "giả sốt", "giả bệnh", "dàn dựng", "giả ngất",
    "giả ốm", "ma ám", "siêu nhiên", "nghiện bọt xà phòng", "dởm",
    "người chưa từng học y", "không cần dùng thuốc",
    "nước tiểu nhân tạo", "làm loãng mẫu", "nước tiểu sạch",
)

NON_MEDICAL_LIFESTYLE_TERMS = term_set(
    "trang điểm", "sơn móng", "tẩy lông", "cạo lông", "uốn tóc", "nhuộm tóc",
    "tạo kiểu tóc", "chụp ảnh", "thời trang", "du lịch", "thú cưng",
    "giảm mỡ", "tăng cơ", "thể hình", "trồng cây", "nuôi cá", "làm đẹp",
    "mỹ phẩm", "serum", "retinol", "nếp nhăn", "làm sáng da",
    "dưỡng da", "dưỡng tóc", "chăm sóc tóc", "tẩy tế bào chết",
    "chống lão hóa", "chống lão hoá", "botox", "chất độn", "căng da",
    "triệt lông", "salon", "thẩm mỹ", "thẩm mỹ viện",
    "kem dưỡng ẩm", "dưỡng ẩm", "kem chống nắng", "đốm đồi mồi",
    "siêu mài mòn", "làm trắng răng", "chuyến xe yêu thương",
    "xe nghĩa tình", "về quê ăn tết", "về quê đón tết", "quà tết",
    "khả năng học", "tăng trí nhớ",
    "nước hoa", "phủ tóc bạc", "trẻ hơn", "cắt tỉa tóc", "bảo vệ tóc",
    "thanh lọc cơ thể", "mùi cơ thể", "chăm sóc da",
)

STRONG_NON_MEDICAL_LIFESTYLE_TERMS = term_set(
    "trang điểm", "sơn móng", "tẩy lông", "cạo lông", "uốn tóc",
    "nhuộm tóc", "tạo kiểu tóc", "thời trang", "làm đẹp", "mỹ phẩm",
    "nếp nhăn", "làm sáng da", "chống lão hóa", "chống lão hoá",
    "botox", "chất độn", "căng da", "triệt lông", "salon", "thẩm mỹ",
    "thẩm mỹ viện", "làm trắng răng", "đốm đồi mồi", "siêu mài mòn",
    "tan mỡ", "đẹp da",
    "cấy râu", "râu mọc", "tăng kích cỡ ngực", "phát triển ngực",
    "làm nâu da", "thải độc",
)

NON_BIOMEDICAL_SOCIAL_TERMS = term_set(
    "chuyến xe", "về quê ăn tết", "về quê đón tết", "ngày hội",
    "ngày quốc tế phụ nữ", "bông hồng", "cắt tóc", "chăm sóc móng",
    "trao quà", "tặng quà", "lễ tri ân", "xuyên việt",
    "đạp xe xuyên việt", "mang lại niềm vui", "xua tan", "từ thiện",
    "quỹ hỗ trợ", "hoàn cảnh khó khăn", "không bỏ học", "nghệ sĩ",
    "ca sĩ", "diễn viên", "mỹ nhân", "chôn sống", "bỏ rơi",
    "nsưt", "nsnd", "dòng status", "tâm thư", "thư khen", "tài xế taxi",
    "đại sứ",
    "món quà", "rộn ràng", "lọt danh sách", "nhà khoa học tiêu biểu",
    "hiến máu tình nguyện", "chu du", "dự án", "tôn vinh", "nghỉ hưu",
    "rời chức", "chia tay", "cầu chúc", "kỳ thi", "ra đi nhưng",
    "mơ ước trở thành", "hiện tượng mạng xã hội", "đăng tải đoạn video",
    "sống đẹp", "năng lượng tích cực", "lớp học yoga", "thiên đàng",
    "kỳ diệu", "cuộc sống vất vả", "sinh nhật", "chuyến đi", "giờ đẹp",
    "thời khắc chuyển giao", "niềm vui sướng", "gây ấn tượng",
    "mạng xã hội", "học trò nghèo", "cố gắng học tập", "trúng tuyển",
    "trên taxi", "tài xế", "clip trên", "hiến máu cứu người",
    "tham gia hiến máu", "chủ nhật đỏ", "tổng thống", "tròn 10 năm",
    "tốt nghiệp", "đạp xe hơn", "tiểu thuyết hư cấu",
    "phim khoa học viễn tưởng",
)


def find_terms(text: str, candidates: Iterable[str]) -> set[str]:
    return {term for term in candidates if f" {term} " in text}


def collect_evidence(text: str) -> dict[str, set[str]]:
    return {
        "human": find_terms(text, HUMAN_TERMS),
        "disease": find_terms(text, DISEASE_TERMS),
        "treatment": find_terms(text, TREATMENT_TERMS),
        "clinical": find_terms(text, CLINICAL_TERMS),
        "health_system": find_terms(text, HEALTH_SYSTEM_TERMS),
        "anatomy": find_terms(text, ANATOMY_TERMS),
        "symptom": find_terms(text, SYMPTOM_TERMS),
        "prevention": find_terms(text, PREVENTION_TERMS),
        "research": find_terms(text, RESEARCH_TERMS),
        "animal": find_terms(text, ANIMAL_TERMS),
        "veterinary": find_terms(text, VETERINARY_TERMS),
        "taxonomy": find_terms(text, TAXONOMY_TERMS),
        "plant": find_terms(text, PLANT_TERMS),
        "incident": find_terms(text, INCIDENT_TERMS),
        "legal_or_crime": find_terms(text, LEGAL_OR_CRIME_TERMS),
        "administrative": find_terms(text, ADMINISTRATIVE_TERMS),
        "strong_administrative": find_terms(text, STRONG_ADMINISTRATIVE_TERMS),
        "deceptive_health": find_terms(text, DECEPTIVE_HEALTH_TERMS),
        "lifestyle": find_terms(text, NON_MEDICAL_LIFESTYLE_TERMS),
        "strong_lifestyle": find_terms(text, STRONG_NON_MEDICAL_LIFESTYLE_TERMS),
        "non_biomedical_social": find_terms(text, NON_BIOMEDICAL_SOCIAL_TERMS),
    }


def medical_score(evidence: dict[str, set[str]]) -> int:
    return (
        5 * len(evidence["disease"])
        + 5 * len(evidence["treatment"])
        + 3 * len(evidence["clinical"])
        + 2 * len(evidence["health_system"])
        + 2 * len(evidence["anatomy"])
        + 2 * len(evidence["symptom"])
        + 2 * len(evidence["prevention"])
        + 2 * len(evidence["research"])
    )


def serialize_evidence(evidence: dict[str, set[str]]) -> str:
    parts = []
    for category, matches in evidence.items():
        if matches:
            parts.append(f"{category}={'|'.join(sorted(matches))}")
    return ";".join(parts)


def classify(article: str, summary: str) -> dict[str, Any]:
    summary_text = normalize_for_matching(summary)
    summary_evidence = collect_evidence(summary_text)
    article_evidence = collect_evidence(normalize_for_matching(article))
    score = medical_score(summary_evidence)
    article_support = min(6, medical_score(article_evidence) // 3)
    score += article_support

    animal = bool(summary_evidence["animal"])
    explicit_human = bool(summary_evidence["human"])
    veterinary = bool(summary_evidence["veterinary"])
    zoology = bool(summary_evidence["animal"] and summary_evidence["taxonomy"])
    plant_care = bool(summary_evidence["plant"])
    generic_treatment = summary_evidence["treatment"] & {"dùng thuốc"}
    specific_treatment = summary_evidence["treatment"] - generic_treatment
    treatment_focus = bool(
        specific_treatment
        or (
            generic_treatment
            and (
                summary_evidence["disease"]
                or summary_evidence["clinical"]
                or summary_evidence["health_system"]
                or summary_evidence["human"]
                or summary_evidence["symptom"]
            )
        )
    )
    medical_core = bool(summary_evidence["disease"] or treatment_focus)
    human_zoonotic_context = bool(
        summary_evidence["disease"] & HUMAN_ZOONOTIC_DISEASE_TERMS
        and (
            explicit_human
            or summary_evidence["clinical"]
            or summary_evidence["health_system"]
            or summary_evidence["symptom"]
        )
    )
    human_exposure_context = bool(
        find_terms(summary_text, HUMAN_EXPOSURE_ANIMAL_TERMS)
        and (
            summary_evidence["treatment"]
            or summary_evidence["clinical"]
            or summary_evidence["health_system"]
            or summary_evidence["symptom"]
        )
    )
    human_medical_context = bool(
        human_zoonotic_context
        or human_exposure_context
        or (
            explicit_human
            and (
                summary_evidence["treatment"]
                or summary_evidence["clinical"]
                or summary_evidence["prevention"]
                or summary_evidence["symptom"]
                or (
                    summary_evidence["disease"]
                    and summary_evidence["health_system"]
                )
            )
        )
    )
    clinical_focus = bool(
        summary_evidence["clinical"]
        and (
            summary_evidence["symptom"]
            or summary_evidence["anatomy"]
            or (explicit_human and len(summary_evidence["clinical"]) >= 2)
        )
    )
    biomedical_research = bool(
        summary_evidence["research"]
        and (
            summary_evidence["disease"]
            or summary_evidence["treatment"]
            or summary_evidence["anatomy"]
        )
    )
    translational_animal_research = bool(
        biomedical_research
        and (
            summary_evidence["disease"]
            or summary_evidence["treatment"]
            or (
                explicit_human
                and summary_evidence["anatomy"] & CORE_RESEARCH_ANATOMY_TERMS
            )
            or (
                article_evidence["human"]
                and summary_evidence["anatomy"] & CORE_RESEARCH_ANATOMY_TERMS
            )
        )
    )
    symptom_focus = bool(
        len(summary_evidence["symptom"]) >= 2
        and not summary_evidence["lifestyle"]
        and not summary_evidence["deceptive_health"]
    )
    healthcare_focus = bool(
        summary_evidence["health_system"]
        and (
            medical_core
            or summary_evidence["clinical"]
            or summary_evidence["symptom"]
            or summary_evidence["prevention"]
            or summary_evidence["research"]
        )
    )
    preventive_health = bool(
        summary_evidence["prevention"]
        and (
            summary_evidence["disease"]
            or summary_evidence["clinical"]
            or summary_evidence["health_system"]
            or summary_evidence["anatomy"]
            or summary_evidence["symptom"]
        )
    )
    human_biology = bool(
        summary_evidence["anatomy"]
        and (
            explicit_human
            or summary_evidence["research"]
            or summary_evidence["health_system"]
            or len(summary_evidence["anatomy"]) >= 2
            or bool(
                summary_evidence["anatomy"]
                & {"âm đạo", "dương vật", "tử cung", "buồng trứng", "tinh hoàn"}
            )
        )
    )

    if veterinary:
        keep, reason = False, "veterinary_content"
    elif zoology:
        keep, reason = False, "zoology_or_ecology"
    elif (
        animal
        and not human_medical_context
        and not translational_animal_research
    ):
        keep, reason = False, "nonhuman_animal_content"
    elif plant_care:
        keep, reason = False, "plant_care"
    elif summary_evidence["non_biomedical_social"]:
        keep, reason = False, "social_or_human_interest_story"
    elif summary_evidence["strong_lifestyle"]:
        keep, reason = False, "nonmedical_lifestyle"
    elif (
        summary_evidence["lifestyle"]
        and not summary_evidence["disease"]
    ):
        keep, reason = False, "nonmedical_lifestyle"
    elif summary_evidence["deceptive_health"]:
        keep, reason = False, "deceptive_or_nonmedical_health_content"
    elif summary_evidence["legal_or_crime"]:
        keep, reason = False, "legal_or_crime_content"
    elif summary_evidence["strong_administrative"]:
        keep, reason = False, "healthcare_administration_only"
    elif (
        summary_evidence["administrative"]
        and not (
            biomedical_research
            or preventive_health
            or (
                summary_evidence["disease"]
                and (
                    clinical_focus
                    or summary_evidence["symptom"]
                    or summary_evidence["prevention"]
                )
            )
        )
    ):
        keep, reason = False, "healthcare_administration_only"
    elif (
        summary_evidence["incident"]
        and not (
            summary_evidence["disease"]
            and specific_treatment
            and clinical_focus
        )
    ):
        keep, reason = False, "incident_without_medical_focus"
    elif medical_core and score >= 7:
        keep, reason = True, "disease_or_treatment"
    elif biomedical_research and score >= 7:
        keep, reason = True, "biomedical_research"
    elif healthcare_focus and score >= 8:
        keep, reason = True, "healthcare_system_or_clinical_care"
    elif preventive_health and score >= 8:
        keep, reason = True, "prevention_or_human_health"
    elif clinical_focus and score >= 8:
        keep, reason = True, "clinical_care"
    elif symptom_focus and score >= 10:
        keep, reason = True, "human_symptoms_or_injury"
    elif human_biology and score >= 8:
        keep, reason = True, "human_biology"
    else:
        keep, reason = False, "insufficient_biomedical_evidence"

    return {
        "keep": keep,
        "reason": reason,
        "score": score,
        "summary_evidence": serialize_evidence(summary_evidence),
        "article_support_score": article_support,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_split_file(input_dir: Path, split: str) -> Path:
    matches = sorted(input_dir.glob(f"{split}_*.*"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {split}_* file in {input_dir}; found {len(matches)}."
        )
    return matches[0]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".csv":
        frame = pd.read_csv(path, keep_default_na=False)
    elif path.suffix.casefold() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {path}")
    if not {"article", "summary"}.issubset(frame.columns):
        raise ValueError(f"Missing article/summary columns in {path}")
    return frame[["article", "summary"]].copy()


def write_bundle(input_dir: Path, output_dir: Path, audit_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if audit_dir.exists() and any(audit_dir.iterdir()):
        raise FileExistsError(f"Audit directory is not empty: {audit_dir}")

    output_parent = output_dir.parent.resolve()
    audit_parent = audit_dir.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    audit_parent.mkdir(parents=True, exist_ok=True)
    output_tmp = Path(tempfile.mkdtemp(prefix=".phase2-biomedical-", dir=output_parent))
    audit_tmp = Path(tempfile.mkdtemp(prefix=".phase2-audit-", dir=audit_parent))

    report: dict[str, Any] = {
        "filter_name": "conservative_human_biomedical_filter_v12",
        "definition": (
            "Human disease, diagnosis, treatment, clinical care, biomedical "
            "research, human biology, prevention, and health systems."
        ),
        "excluded_domains": [
            "veterinary care",
            "zoology and ecology",
            "plant care",
            "non-medical lifestyle",
            "incidents without a medical focus",
            "legal or crime stories",
            "healthcare administration without biomedical content",
            "social and human-interest stories",
        ],
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "audit_dir": str(audit_dir.resolve()),
        "splits": {},
    }
    excluded_frames: list[pd.DataFrame] = []

    try:
        for split in SPLITS:
            input_path = find_split_file(input_dir, split)
            frame = read_table(input_path)
            decisions = pd.DataFrame(
                [
                    classify(str(row.article), str(row.summary))
                    for row in frame.itertuples(index=False)
                ]
            )
            kept = frame.loc[decisions["keep"]].reset_index(drop=True)
            excluded = frame.loc[~decisions["keep"]].reset_index(drop=True)
            excluded_decisions = decisions.loc[~decisions["keep"]].reset_index(
                drop=True
            )
            excluded = pd.concat([excluded_decisions, excluded], axis=1)
            excluded.insert(0, "split", split)
            excluded_frames.append(excluded)

            output_path = output_tmp / f"{split}_1.csv"
            kept.to_csv(output_path, index=False)
            report["splits"][split] = {
                "input_file": str(input_path.resolve()),
                "input_sha256": sha256_file(input_path),
                "input_rows": len(frame),
                "kept_rows": len(kept),
                "excluded_rows": len(excluded),
                "keep_rate": round(len(kept) / len(frame), 6),
                "kept_reason_counts": {
                    str(reason): int(count)
                    for reason, count in sorted(
                        decisions.loc[
                            decisions["keep"], "reason"
                        ].value_counts().items()
                    )
                },
                "excluded_reason_counts": {
                    str(reason): int(count)
                    for reason, count in sorted(
                        decisions.loc[
                            ~decisions["keep"], "reason"
                        ].value_counts().items()
                    )
                },
            }

        excluded_all = pd.concat(excluded_frames, ignore_index=True)
        excluded_all.to_csv(audit_tmp / "excluded_non_biomedical.csv", index=False)

        public_frames = {
            split: pd.read_csv(output_tmp / f"{split}_1.csv", keep_default_na=False)
            for split in SPLITS
        }
        article_sets = {
            split: set(frame["article"].map(normalize_for_matching))
            for split, frame in public_frames.items()
        }
        overlaps = {
            "train_validation": len(article_sets["train"] & article_sets["validation"]),
            "train_test": len(article_sets["train"] & article_sets["test"]),
            "validation_test": len(
                article_sets["validation"] & article_sets["test"]
            ),
        }
        report["validation"] = {
            "empty_article_rows": int(sum(
                frame["article"].astype(str).str.strip().eq("").sum()
                for frame in public_frames.values()
            )),
            "empty_summary_rows": int(sum(
                frame["summary"].astype(str).str.strip().eq("").sum()
                for frame in public_frames.values()
            )),
            "exact_duplicate_pairs": int(sum(
                frame.duplicated(["article", "summary"]).sum()
                for frame in public_frames.values()
            )),
            "cross_split_article_overlaps": overlaps,
        }
        report["totals"] = {
            "input_rows": sum(item["input_rows"] for item in report["splits"].values()),
            "kept_rows": sum(item["kept_rows"] for item in report["splits"].values()),
            "excluded_rows": sum(
                item["excluded_rows"] for item in report["splits"].values()
            ),
            "excluded_reason_counts": dict(
                sorted(Counter(excluded_all["reason"]).items())
            ),
        }
        (audit_tmp / "biomedical_filter_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if output_dir.exists():
            output_dir.rmdir()
        if audit_dir.exists():
            audit_dir.rmdir()
        os.replace(output_tmp, output_dir)
        os.replace(audit_tmp, audit_dir)
        return report
    except Exception:
        shutil.rmtree(output_tmp, ignore_errors=True)
        shutil.rmtree(audit_tmp, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter cleaned Phase 2 data to the human biomedical domain."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/phase_2"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/phase_2_biomedical")
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("data/phase_2_biomedical_audit")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = write_bundle(args.input_dir, args.output_dir, args.audit_dir)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
