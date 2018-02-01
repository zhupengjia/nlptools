#!/usr/bin/env python3
from ailab.text.tokenizer import Segment_Simple
from ailab.utils import zload
import sys

cfg = {'TOKENIZER':'simple', 'stopwords_path':'stopwords.txt'}
s = Segment_Simple(cfg)
#print(s.seg(sys.argv[1]))

txt = '''bSoapUtil.java:68) ERROR com.asiainfo.scrm.base.common.soap.EsbSoapUtil - [NOT ERROR] ESB Output==>[ESB_CS_QRY_RECORD_MULTIBILL_001]<?xml version='1.0' encoding='UTF-8'?><RESP_PARAM><PUB_INFO><RETURN_RESULT>0</RETURN_RESULT><RETURN_DESC>操作成功</RETURN_DESC><OSB_SERIAL_NO>b-app-q19-srv08^15159352570000000026</OSB_SERIAL_NO></PUB_INFO><BUSI_INFO><CONTENT>&lt;?xml version="1.0" encoding="GBK" ?&gt;&lt;对账单&gt;&lt;辅助项&gt;&lt;账单类型&gt;标准&lt;/账单类型&gt;&lt;套餐内科目集&gt;&lt;/套餐内科目集&gt;&lt;优惠科目&gt;&lt;一级项&gt;&lt;项目ID&gt;470&lt;/项目ID&gt;&lt;/一级项&gt;&lt;/优惠科目&gt;&lt;消费科目&gt;&lt;一级项&gt;&lt;项目ID&gt;410&lt;/项目ID&gt;&lt;/一级项&gt;&lt;一级项&gt;&lt;项目ID&gt;450&lt;/项目ID&gt;&lt;/一级项&gt;&lt;一级项&gt;&lt;项目ID&gt;490&lt;/项目ID&gt;&lt;/一级项&gt;&lt;一级项&gt;&lt;项目ID&gt;480&lt;/ 项目ID&gt;&lt;/一级项&gt;&lt;一级项&gt;&lt;项目ID&gt;420&lt;/项目ID&gt;&lt;/一级项&gt;&lt;一级项&gt;&lt;项目ID&gt;430&lt;/项目ID&gt;&lt;/一级项&gt;&lt;一级项&gt;&lt;项目ID&gt;460&lt;/项目ID&gt;&lt;/一级项&gt;&lt;/消费科目&gt;&lt;/辅助项&gt;&lt;标题&gt;中国移动通信 客户账单&lt;/标题&gt;&lt;首页&gt;&lt; 表头区&gt;&lt;客户品牌&gt;全球通&lt;/客户品牌&gt;&lt;客户姓名&gt;周~~&lt;/客户姓名&gt;&lt;客户号码&gt;187****6045&lt;/客户号码&gt;&lt;套餐名称&gt;4G飞享套餐&lt;/套餐名称&gt;&lt;套餐描述&gt;套餐内含:用户通过选择指定“飞享套餐基本模组”和“手机上网模组”（注1），个性化定义套餐内容。套餐赠送来电显示，享受国内被叫免费，超出后国内主叫长市漫享受一口价资费（注2）、上网流量0.29元/MB(不足1M精确到分)。'''

print(s.seg(txt))


