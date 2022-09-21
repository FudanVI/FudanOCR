import torch
from data.lmdbReader import lmdbDataset, resizeNormalize
from config import config
import os
import shutil
from shutil import copyfile
import Levenshtein
import pickle as pkl
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import math
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot


mse_loss = nn.MSELoss()

rad_to_4num_tab = {}
stroke_num_in_radical_file = open('./data/stroke_orientation_num.txt','r',encoding='utf-8')
for line in stroke_num_in_radical_file.readlines():
    elem_list = line.split(',')
    rad_to_4num_tab[elem_list[0]] = [int(elem_list[1]),int(elem_list[2]),int(elem_list[3]),int(elem_list[4])]

word_3755 = '啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆八疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽辊滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰奸缄茧检柬碱硷拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利傈例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐恰洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听烃汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主著柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐柞做作坐座$'
ctw_word = '誓迭意祺厮锚红罹慌茹拷楼叉猎腺兽靠完发真树畸宿陽嫩适慈態初劳上珊寸骗威麵木纳鲜蜜泵掩隅挚粥孕谢晓宸垠凤焖娆撕梧冈代凳雏屿崋略幻认嗨籣滙益熔劵枼浆拿纸龙厉桩焚许众低營藻芸赤轰瓯農嘉洼贵叹刘腰治刑墩硕针狗反運讯陪榈犹卒穴鍋阀应围汇饱纫义開秉都謀豆何翳找栓岚埸逢處丈仃傻司闷空抚裂秒明粗繹显雨倍光戒田论篇撷靜羯矩裱示货淼灵圓钓址骂腔魂即門轻皮沛葆茜郊讲浸丘封夺簧珍战泌澋腚駿邑眯习望抱诗膠朴画逐琐回閣言荟豌碟腋佰媲廉尘去纵纺蚌歷凛海蓄除镗產緩须癜熏逅殷药株伪模契辖屯爱絨惜机炒蜗懋返雷礼兼狮珠伞邹贪鸟延绥栈拟鷗核假腕洪矗还脸缴京喀贴曾旋樣溝創轴腻杭晚锺做驿伶助檀厰训并杆灯铅茸舌谨俗願小陌税纯连择蚝碼豐沥邀将头肤铨阜分宏全坤劉崔荡米巅强盧茅龟否喽耕莅申陇礴充精暉公考酵避髓道鰲移活報犯黄失汾扇党計析順镶吹奈馒裡答收称坎愿燕叶葫舶俏癸虚祐淅貌仅購惟呈肖紡欢短钙病持划崂闵贡矢樾脖辟解朕赵控軒拱影剿项荐华濱帅八魔峭为靛车与硒虬绸孔蜘账约荤甸在柏昕频製野彌徐麒視熱唐占徳懒束跳拐害起玄侗忧渣切遨舱梭陆湃稼然阅缕薬烛隨黑泸随茵蚕菁簘尝脂配涂绝沁截吊瓷五欲颍征扎炖闽建鹰庙顶口科肽裝博穗味蘭沏您冻縁酷眀氧首姆澄开沙兢策样钨方尬棉禽億溪铠左借塱兴教莘宣铁呷锈鎮庸覺奖撈鸣到波菌原渡未帮缪检偉蚬僧偶险而耳氽锌艰父滞忙纬斋衬觅彰渔屎皇曲飮作慎飛浙廓覚華陕蘇骄豚盐臺割峰弯垒标徒下曦拌萄史西学雍泳妈頁桶养题锐怕惕销嫣璀碳擦产荊骊筑膨兜池油亩垂馈们癫臊必秦郎尿谛旁歇员冀近廠啸戴李绚掬藩总救汀蜒播片淇祖触暑衡珺次什侬清菱干食织玛佛纠匠尕閤涌箂鹿胖替蝶仟业畫爆疙鹅包受音邯撞阵輕云绕例索绪玮铺需媒按革告喬裘訪軸忠領泓詳觳逊费種壽氏妥拖璎昌胶臭嫁老犬暮哺澈诱础促扔汗诈卡豹晖葛绅皓峪态定改崛冉报仰慕绞廳栗湾页阔倾丝住丰耗哨浑睿碗审绿企遠刹凉肝超式昂洲液澳绮督罩梵铜厌學肆戏志富悔同茗累额休緹芝嗜也佳欧溧数迦艮枪杖型严昇婴赖确韵坦背昨搬取筷己莲葚紅具甲膳权孤桥泉古磐鞋圃鉑鹫手放康瀑螺嘟元譜网疲雁霜衙塑旯鴐帆币吞研灭从琦霄甩尧晌饮兵函姚冒唤敢赌较润底翅埔炸勁焊蓬勘芽册嬉動港桨罐耀庐蓟儲莼友荫瑰溶弘涮足芊盆我粉蕉璜净享輪榄玉軟復横斯稻练释诀寒朽羿级之車偭蕭钜棱蚊金粱聚群當副斤幺途钰兒茂嶺竿赐举系軍氡架婚饺寰终幸柄玻牵不现釉湯斗酝济傷跨妹诉橙可梅能钎禅折迠焦辱揭浣烯夹塔裁哆字军瞿啼鸥築傳舘瑞范沧声噪情秘漠青负浓蔽桑董俩聆焰疚邺菇各碑鳥枇吳维萌眸校栅牌继昔衞草罕聊闹氩環鎖入钻颐你麺肚募鴻匾力类梦驴白酸瘩喝蜻鸽觀夸葉阳射造陳性缸瀘炳玖姑罄引敦立攀皙赠碱据间秩萃多年盈芬橘韶驭歳芷筹裕乐斑圜托盾钧问魚涧圗鼓珲渎廖菜萦裹牦喫煌娲芦尖贯照赔站慶件亚汛吟酪荞憨卉肿顷會卓更零鄂琪倦萂彻靖跌潼阑扩径皆着患慰优雄映伦淋里罗雪台廈媛措厨括付誉亲羔砚崖黎眷瓢恰郸电艺没升月戈杨舍盎测场呱辈佣密腾擀魏松赂坛釜矫锁連蒙沟娃來夀驶棚師递聲嵩濠親轨扪鄭夕錢哗聨靶省眺芳盟淀泾吏右难廂文肢攻稽梯程塘典就奢九骨竞詩熟租哦效兮证是襲沐缤她巡羞霓帽樂团扣妍层澐稚晨喊附果合髪章晰棕牧妲独两过淑陀祭成崇细愧虎陡商匹枕郑任辐免壘锣渴驲污祸朔县聪哑绗歌议奉筆際委肩轿村鑫蛋掏似菧袖瑷诤腊矿戎肠品禹煕伽毯鞍钊脆号奎烘跶夌旨沂芋量垃讨禺壳林枢念残彩施二弟献怠前日屡醉恪纱痧萝城辅荆眠枯暖團请些保恤钛稳看泽梨胆经蒿飞筋笃咏亞務勋侵旧园粒織隍鼠壶茄吋编齐隔桂门硬冰德鹧器逃办恒室咸镜聫钵節煎宛支薪路仔懂财淄又慢扬喜刷題殊疾预廊柚傢专樓炉误胑笋很磁寝篮呼试谈現濛泪晏纶興素楽霞气莉扁窩滚允煮飲莺疝菊诊悦拒泗部私蛮腐突七如遇澤最毅琍歡葵偷衰让猫状椎誡王躺段捐倡幼忘賢胃漫霖宇番絲傑邮瓶毕舰扶瑚愁賣尚腸肥深羊亦姓啃乾拆逸锋叫晴宾鶏输腑染炅万拍澡汪蛇速挥母血绣葬钲哥孩鴿泛挖损烁茉储烫鄕甬帝破谁三子遍帐角悬吃它迁其闯串地咪渠鸭桦签语错奠諧廎协距锲浦川穩陋亏郁潮睡君熙鳌瘪衔僅扭通摸援乘心虫爪靡杰砺航南畵律尾暂布睫肴领循榻襄漾指辋阁祛弹芜蔓和旬飙另蹄扰漂贝迹东堰瑜瓦昊辆嶪藕銀咨象滇檬竖贰疯绘带邂妻煤船节歲牛禮莊衣臣赞直巢骆鷄溢急淮招档琢剧耻氯鬼癣楠桁愈体击貨鬥餐臨吗卧弥览倪自筝雹範妇砼浮椿習消详驱骐頭械虽仓劫苑谦蔡导良排杯圩厚椒壁息撼為魁单菩廛亢蜀槐躲兆萼接者柴土把杀歧壮停葡儿谋緣句仪十跃操临潞客早痘羙脱砾已缘凖役摄缎轮推娅濮究位混醇惬断蟹叁蜂汽浅守运洋绵烦伸酒吸决棵镀远樨卫蟠婆窖旦刺秀沫暨岸腌僖组變铸居给砖丫图鸪迩宪摘誠籬蕾丹荘琳伙寄追露掉疫理纲養飚积奋条孚驾忽娱锡逼疑要辛率練薄危旅岭钥浩新俬鲍給阖疼颢承茨且靚盘共峻凯兄踪朵刻满儒縣熬協藏默夢饿麥氙義悼玩棠訓饰呆恕盒炴障炬努视粘案泰褥鲤穿诏盔吨鞭脑楚登担庵贤軽昏質剂命滩故叽骇玺賓漁天幢痣暄炝抛长拓恋區购事緒搞吾贸钣衢磊龄克舒桃姜屏漕杠卢氛酗买翎恐機厢勿票栋框滾珞寇款炮兩再璟軰縂纪鐘姿坂肯爲链彧從毡貝備咽库英苯则坐蛟参拇鵝餃堵谱沱投鼻宴閏俞壬國蹈袁冬囱星遗灼农抢嫂幽樽贞孢勝动堅艾潇风瓜剃仑丧倒捕凌先脊算锂癌罪娘添来男予淘陈概個一診录汤筒袂盗氟韦薡灡补灾暴烂传與埠酿面潜齿馍謝聯籍序弱撮展廣簾胞鉴庭鳮銷肉春羚跑渗匮棘疏流邻翟像逝輝瘀贫囊琥燎悠街邦鍍汝侨弓佑桔嵗苟轧苦尊竟隐豫验匙窃弩调若晟太所店誼橄蚁蒲宜弈工藝潔汉洞被于硫振疗涵银记涛甫均履赏局醒問师撒护酱进见香俄佐民乎槽搜惩半椴彭靓冲辽选伐歹质阴股讼会好家钉磅敖戛价膜霍宅趣败高胎赁火偿黛甄软仗叔瀚装罚伏盛曹惰季榆畅丛知岛潘擅猴鐵闻邵掀扫麓访仲挂傣鹏娄韬雾殡耽陵吖崧尺調帛疮亿撤安闭彊榜拎垦驳颜妮贺浇午压雅几寨琼驰扒寿废善漆痛侯恢列巩蕴昼觉佩薯甘乔识螂普湘坞吐瑗疆璐花遊快剪嵊吵岳沉的歆悉將蟑迅衫箍芭翌饲仁醸历螃观对乃旮恩蹓置胗設闌宮胥灿痕枫等降沈丁蔚医炫糖骏哈启晋碎裤袭窝瓣闲仕阿雲烨烤眾说奥读名简滨巨沃喧眙炭還侈盖线槁焗禾崴采扦驻久墙孝翰粮點岁含馄嘻炊载冶属圈抄归憶滤啡薇套座只况圭向蒋缠芙贷負爸雞洗少毛鋼楞娠澎継违掌窄诺孟琅歴令及信搏咛拉燃闪狂倩楓惠钢绑警皂語坏散浜猛袋厂尔挽士夫饨杉辞甜缆堂炔艳极使窗千娟挑劲沌經妙邱妃烩止填抓表偢健生禦漪餠畏張祠莓菓雯余耘貢旭瞻圖当費磨塲稅乳奧留丑际书童銭咫培摩晶馨兰豪吴值颂利莱菠鲸瞩拔贿酥氣争屹狼顾窑境財基宫笛欣囧尤述闫晒易坪鄉褡隱铣此狠津碧北旺熄椰栖尹闸押侣中浒灸沪璨炼哇犇巫苔寻鱼内度霸奕哉芯脚离广長匡饶漢致馅憾傲厅达請目诸诚拨渚感岔席滁鉄待烟璧府喉牙由叙迪山釘蚂滴蝠扈亨葩骚缝袍敗材透规懈扑則术勇锯整鸡怡帕奶美倬床咯或裙般艇沾乡提勃婵脏磚笔转濟谓缺渍准宗译管格俶礫洽潤注霆赚巾黏凰挤內劝监授舟踩每權坝存鳯奏榭埋创粹牢鲁愛爬设氮化苗毁侥禧墨杷渤结死印荒夏嗑呵焕彤摇勞铭姨官雙證堡福脉迈鹤巧腩出歉搭哟塞肛盼禁挣勤麦缇伯犟灌坡俯晤菲曼央蜡拜森柜媚评育蜕嫌孙橋缔抹宠凇紧侧刊啦淡常颈臻龍漏擎橱惊亮伍稞唯奴覽龈狐铝希畔饼暗婷梓繁缅場糙哲翁柳综眼榨绽欺奇颗劈神黒爵块憩訂本莹坟肇茶粽辣抬版平陶飯板羅菀礻呐查咕尴帜么桌江總咱增囍他秋症恬马形差塌墅堤湛柱邨蕐硚稀桷珑后大紀浴卜姗便涨丽莎敏坷瞳削庞苏痤携俭悟汁伴六仆潭宋擒鋁饭橡靴昆因岩正往遵樞扛铂鋪曙功派浏铍屋瞬盅探估双融莭舜餅栏楹駕绎惯卖盏国份炎堆赣琊词枣鱿社舫羽逛匯萧皱庆凝洱仿璃穆技伊琨間咖虹莫市踏約刃拳候遮澜顿弄婕遣抗忆顺娜泡制涞兑想陷舞袜弧槟哪衷翼今诞吒煨乓倉隆谅售書四插求瘦召嘿峨垫琉冥缩荷竺嘘峡祈涉骅溜喱怀汕摊卤樟百棋屬资虾坚赈蛛笼铬吕邊御寶写身势雕巴尽沿泼寞様送绒埭覆域券墓责务耸井瑶防啄朗鮮蓝谷槓丸馋源髮比滑坊绩赶勒婧棰醋挞丶复迷鈴加貴颁赋暧洮换阻涿幕烈姐體聘詠晾貂畜殓笑錦強藤翡刀昱瑛计圳世刨廷啧女涡滥猪主胡熊痹俱询電队妩坑法尼描潢翻点邸柔喂得察帘朋瘤睛周杜第筱圆肾演仙捍钟贻祝灰發飓禄聖啊却醫话妊麟馬蝇唱粵雀兿服赫震涟实房续才翠苹葱棺鏡联試异但毫殿营塵有馆灶驼苼捷虞鸿励怎楊肌洄蒸揽彼惑風瘾濃毗辰飘阶耐珂时笕達雜步竹匝東钩術伟拾乱容鲢减瑙進驽祥兔皋汐薹樱打河乌乙够炙码环端岗瑟构蟲劍椅抵啥毂賀慧廚麯嚣寺幅渝水桐行末宝裢毒邪瑪筛永焙邓庄裳思呢跆糊柯硅泥纷亭镇旱奔备边握区搅館始笈涤贩褪蕓晦蔻执用厦诂恭挡舵柑婦凡鹭旗紫走石維猕寳璞乒修齊隧户壹圾醬旷揚頂丙玲窨蛙蝎集交吉纽腿洁荣该篷落冠关职缓狱滋谭駛武煲欠徽臂巳胀淞铖杂溯拥锦那缦柠湖碍竣樊球焱秤烧梁族聂憬專斌坯渊朱叠菡記悲响藍佬抽物批温纹激糕杏厘宽变芒静嘴著僑期逆號践屈越既获铃界逹勺騏钞符赢刚凔娇郴業鼎統丨嗒扳人賞订麻逗根键特谐卷冷酬沸疤这豊竭苍依恶处料叮班皖匣胸伤粀逻相砂种兹曝髙寓妆限珀魅逺赛煜无萨肃實萬巷掘殖無藥遥銹虑宁凭凱政牡肘退飾拼锅叱湿胜萱閑辉氢巍绳景箭孜纤宵热萍介轩植幔互洛鄞了骑棒箱喔時籁别産阱至蓮非穷涯弗困敬钠蝈智愉盱啫逾捆麗剑听傅枝睦碰迎蔬浪韩弃粤届络貿个啓固翔網绍蒂眉圣榔刮戶郢島爷勾蓉馕鳳園张爽腹郭韭玫钱瑯俊供外重课色厕茏朝谊吧郡跟匀偏翱州啤荔臆以膏盲泊參淳蜓碚琴燥捞玓話统院榕鵔微游夜斜嶩喷'
if config['alphabet'] == 'ctw':
    alphabet_character = '<誓迭意祺厮锚红罹慌茹拷楼叉猎腺兽靠完发真树畸宿陽嫩适慈態初劳上珊寸骗威麵木纳鲜蜜泵掩隅挚粥孕谢晓宸垠凤焖娆撕梧冈代凳雏屿崋略幻认嗨籣滙益熔劵枼浆拿纸龙厉桩焚许众低營藻芸赤轰瓯農嘉洼贵叹刘腰治刑墩硕针狗反運讯陪榈犹卒穴鍋阀应围汇饱纫义開秉都謀豆何翳找栓岚埸逢處丈仃傻司闷空抚裂秒明粗繹显雨倍光戒田论篇撷靜羯矩裱示货淼灵圓钓址骂腔魂即門轻皮沛葆茜郊讲浸丘封夺簧珍战泌澋腚駿邑眯习望抱诗膠朴画逐琐回閣言荟豌碟腋佰媲廉尘去纵纺蚌歷凛海蓄除镗產緩须癜熏逅殷药株伪模契辖屯爱絨惜机炒蜗懋返雷礼兼狮珠伞邹贪鸟延绥栈拟鷗核假腕洪矗还脸缴京喀贴曾旋樣溝創轴腻杭晚锺做驿伶助檀厰训并杆灯铅茸舌谨俗願小陌税纯连择蚝碼豐沥邀将头肤铨阜分宏全坤劉崔荡米巅强盧茅龟否喽耕莅申陇礴充精暉公考酵避髓道鰲移活報犯黄失汾扇党計析順镶吹奈馒裡答收称坎愿燕叶葫舶俏癸虚祐淅貌仅購惟呈肖紡欢短钙病持划崂闵贡矢樾脖辟解朕赵控軒拱影剿项荐华濱帅八魔峭为靛车与硒虬绸孔蜘账约荤甸在柏昕频製野彌徐麒視熱唐占徳懒束跳拐害起玄侗忧渣切遨舱梭陆湃稼然阅缕薬烛隨黑泸随茵蚕菁簘尝脂配涂绝沁截吊瓷五欲颍征扎炖闽建鹰庙顶口科肽裝博穗味蘭沏您冻縁酷眀氧首姆澄开沙兢策样钨方尬棉禽億溪铠左借塱兴教莘宣铁呷锈鎮庸覺奖撈鸣到波菌原渡未帮缪检偉蚬僧偶险而耳氽锌艰父滞忙纬斋衬觅彰渔屎皇曲飮作慎飛浙廓覚華陕蘇骄豚盐臺割峰弯垒标徒下曦拌萄史西学雍泳妈頁桶养题锐怕惕销嫣璀碳擦产荊骊筑膨兜池油亩垂馈们癫臊必秦郎尿谛旁歇员冀近廠啸戴李绚掬藩总救汀蜒播片淇祖触暑衡珺次什侬清菱干食织玛佛纠匠尕閤涌箂鹿胖替蝶仟业畫爆疙鹅包受音邯撞阵輕云绕例索绪玮铺需媒按革告喬裘訪軸忠領泓詳觳逊费種壽氏妥拖璎昌胶臭嫁老犬暮哺澈诱础促扔汗诈卡豹晖葛绅皓峪态定改崛冉报仰慕绞廳栗湾页阔倾丝住丰耗哨浑睿碗审绿企遠刹凉肝超式昂洲液澳绮督罩梵铜厌學肆戏志富悔同茗累额休緹芝嗜也佳欧溧数迦艮枪杖型严昇婴赖确韵坦背昨搬取筷己莲葚紅具甲膳权孤桥泉古磐鞋圃鉑鹫手放康瀑螺嘟元譜网疲雁霜衙塑旯鴐帆币吞研灭从琦霄甩尧晌饮兵函姚冒唤敢赌较润底翅埔炸勁焊蓬勘芽册嬉動港桨罐耀庐蓟儲莼友荫瑰溶弘涮足芊盆我粉蕉璜净享輪榄玉軟復横斯稻练释诀寒朽羿级之車偭蕭钜棱蚊金粱聚群當副斤幺途钰兒茂嶺竿赐举系軍氡架婚饺寰终幸柄玻牵不现釉湯斗酝济傷跨妹诉橙可梅能钎禅折迠焦辱揭浣烯夹塔裁哆字军瞿啼鸥築傳舘瑞范沧声噪情秘漠青负浓蔽桑董俩聆焰疚邺菇各碑鳥枇吳维萌眸校栅牌继昔衞草罕聊闹氩環鎖入钻颐你麺肚募鴻匾力类梦驴白酸瘩喝蜻鸽觀夸葉阳射造陳性缸瀘炳玖姑罄引敦立攀皙赠碱据间秩萃多年盈芬橘韶驭歳芷筹裕乐斑圜托盾钧问魚涧圗鼓珲渎廖菜萦裹牦喫煌娲芦尖贯照赔站慶件亚汛吟酪荞憨卉肿顷會卓更零鄂琪倦萂彻靖跌潼阑扩径皆着患慰优雄映伦淋里罗雪台廈媛措厨括付誉亲羔砚崖黎眷瓢恰郸电艺没升月戈杨舍盎测场呱辈佣密腾擀魏松赂坛釜矫锁連蒙沟娃來夀驶棚師递聲嵩濠親轨扪鄭夕錢哗聨靶省眺芳盟淀泾吏右难廂文肢攻稽梯程塘典就奢九骨竞詩熟租哦效兮证是襲沐缤她巡羞霓帽樂团扣妍层澐稚晨喊附果合髪章晰棕牧妲独两过淑陀祭成崇细愧虎陡商匹枕郑任辐免壘锣渴驲污祸朔县聪哑绗歌议奉筆際委肩轿村鑫蛋掏似菧袖瑷诤腊矿戎肠品禹煕伽毯鞍钊脆号奎烘跶夌旨沂芋量垃讨禺壳林枢念残彩施二弟献怠前日屡醉恪纱痧萝城辅荆眠枯暖團请些保恤钛稳看泽梨胆经蒿飞筋笃咏亞務勋侵旧园粒織隍鼠壶茄吋编齐隔桂门硬冰德鹧器逃办恒室咸镜聫钵節煎宛支薪路仔懂财淄又慢扬喜刷題殊疾预廊柚傢专樓炉误胑笋很磁寝篮呼试谈現濛泪晏纶興素楽霞气莉扁窩滚允煮飲莺疝菊诊悦拒泗部私蛮腐突七如遇澤最毅琍歡葵偷衰让猫状椎誡王躺段捐倡幼忘賢胃漫霖宇番絲傑邮瓶毕舰扶瑚愁賣尚腸肥深羊亦姓啃乾拆逸锋叫晴宾鶏输腑染炅万拍澡汪蛇速挥母血绣葬钲哥孩鴿泛挖损烁茉储烫鄕甬帝破谁三子遍帐角悬吃它迁其闯串地咪渠鸭桦签语错奠諧廎协距锲浦川穩陋亏郁潮睡君熙鳌瘪衔僅扭通摸援乘心虫爪靡杰砺航南畵律尾暂布睫肴领循榻襄漾指辋阁祛弹芜蔓和旬飙另蹄扰漂贝迹东堰瑜瓦昊辆嶪藕銀咨象滇檬竖贰疯绘带邂妻煤船节歲牛禮莊衣臣赞直巢骆鷄溢急淮招档琢剧耻氯鬼癣楠桁愈体击貨鬥餐臨吗卧弥览倪自筝雹範妇砼浮椿習消详驱骐頭械虽仓劫苑谦蔡导良排杯圩厚椒壁息撼為魁单菩廛亢蜀槐躲兆萼接者柴土把杀歧壮停葡儿谋緣句仪十跃操临潞客早痘羙脱砾已缘凖役摄缎轮推娅濮究位混醇惬断蟹叁蜂汽浅守运洋绵烦伸酒吸决棵镀远樨卫蟠婆窖旦刺秀沫暨岸腌僖组變铸居给砖丫图鸪迩宪摘誠籬蕾丹荘琳伙寄追露掉疫理纲養飚积奋条孚驾忽娱锡逼疑要辛率練薄危旅岭钥浩新俬鲍給阖疼颢承茨且靚盘共峻凯兄踪朵刻满儒縣熬協藏默夢饿麥氙義悼玩棠訓饰呆恕盒炴障炬努视粘案泰褥鲤穿诏盔吨鞭脑楚登担庵贤軽昏質剂命滩故叽骇玺賓漁天幢痣暄炝抛长拓恋區购事緒搞吾贸钣衢磊龄克舒桃姜屏漕杠卢氛酗买翎恐機厢勿票栋框滾珞寇款炮兩再璟軰縂纪鐘姿坂肯爲链彧從毡貝備咽库英苯则坐蛟参拇鵝餃堵谱沱投鼻宴閏俞壬國蹈袁冬囱星遗灼农抢嫂幽樽贞孢勝动堅艾潇风瓜剃仑丧倒捕凌先脊算锂癌罪娘添来男予淘陈概個一診录汤筒袂盗氟韦薡灡补灾暴烂传與埠酿面潜齿馍謝聯籍序弱撮展廣簾胞鉴庭鳮銷肉春羚跑渗匮棘疏流邻翟像逝輝瘀贫囊琥燎悠街邦鍍汝侨弓佑桔嵗苟轧苦尊竟隐豫验匙窃弩调若晟太所店誼橄蚁蒲宜弈工藝潔汉洞被于硫振疗涵银记涛甫均履赏局醒問师撒护酱进见香俄佐民乎槽搜惩半椴彭靓冲辽选伐歹质阴股讼会好家钉磅敖戛价膜霍宅趣败高胎赁火偿黛甄软仗叔瀚装罚伏盛曹惰季榆畅丛知岛潘擅猴鐵闻邵掀扫麓访仲挂傣鹏娄韬雾殡耽陵吖崧尺調帛疮亿撤安闭彊榜拎垦驳颜妮贺浇午压雅几寨琼驰扒寿废善漆痛侯恢列巩蕴昼觉佩薯甘乔识螂普湘坞吐瑗疆璐花遊快剪嵊吵岳沉的歆悉將蟑迅衫箍芭翌饲仁醸历螃观对乃旮恩蹓置胗設闌宮胥灿痕枫等降沈丁蔚医炫糖骏哈启晋碎裤袭窝瓣闲仕阿雲烨烤眾说奥读名简滨巨沃喧眙炭還侈盖线槁焗禾崴采扦驻久墙孝翰粮點岁含馄嘻炊载冶属圈抄归憶滤啡薇套座只况圭向蒋缠芙贷負爸雞洗少毛鋼楞娠澎継违掌窄诺孟琅歴令及信搏咛拉燃闪狂倩楓惠钢绑警皂語坏散浜猛袋厂尔挽士夫饨杉辞甜缆堂炔艳极使窗千娟挑劲沌經妙邱妃烩止填抓表偢健生禦漪餠畏張祠莓菓雯余耘貢旭瞻圖当費磨塲稅乳奧留丑际书童銭咫培摩晶馨兰豪吴值颂利莱菠鲸瞩拔贿酥氣争屹狼顾窑境財基宫笛欣囧尤述闫晒易坪鄉褡隱铣此狠津碧北旺熄椰栖尹闸押侣中浒灸沪璨炼哇犇巫苔寻鱼内度霸奕哉芯脚离广長匡饶漢致馅憾傲厅达請目诸诚拨渚感岔席滁鉄待烟璧府喉牙由叙迪山釘蚂滴蝠扈亨葩骚缝袍敗材透规懈扑則术勇锯整鸡怡帕奶美倬床咯或裙般艇沾乡提勃婵脏磚笔转濟谓缺渍准宗译管格俶礫洽潤注霆赚巾黏凰挤內劝监授舟踩每權坝存鳯奏榭埋创粹牢鲁愛爬设氮化苗毁侥禧墨杷渤结死印荒夏嗑呵焕彤摇勞铭姨官雙證堡福脉迈鹤巧腩出歉搭哟塞肛盼禁挣勤麦缇伯犟灌坡俯晤菲曼央蜡拜森柜媚评育蜕嫌孙橋缔抹宠凇紧侧刊啦淡常颈臻龍漏擎橱惊亮伍稞唯奴覽龈狐铝希畔饼暗婷梓繁缅場糙哲翁柳综眼榨绽欺奇颗劈神黒爵块憩訂本莹坟肇茶粽辣抬版平陶飯板羅菀礻呐查咕尴帜么桌江總咱增囍他秋症恬马形差塌墅堤湛柱邨蕐硚稀桷珑后大紀浴卜姗便涨丽莎敏坷瞳削庞苏痤携俭悟汁伴六仆潭宋擒鋁饭橡靴昆因岩正往遵樞扛铂鋪曙功派浏铍屋瞬盅探估双融莭舜餅栏楹駕绎惯卖盏国份炎堆赣琊词枣鱿社舫羽逛匯萧皱庆凝洱仿璃穆技伊琨間咖虹莫市踏約刃拳候遮澜顿弄婕遣抗忆顺娜泡制涞兑想陷舞袜弧槟哪衷翼今诞吒煨乓倉隆谅售書四插求瘦召嘿峨垫琉冥缩荷竺嘘峡祈涉骅溜喱怀汕摊卤樟百棋屬资虾坚赈蛛笼铬吕邊御寶写身势雕巴尽沿泼寞様送绒埭覆域券墓责务耸井瑶防啄朗鮮蓝谷槓丸馋源髮比滑坊绩赶勒婧棰醋挞丶复迷鈴加貴颁赋暧洮换阻涿幕烈姐體聘詠晾貂畜殓笑錦強藤翡刀昱瑛计圳世刨廷啧女涡滥猪主胡熊痹俱询電队妩坑法尼描潢翻点邸柔喂得察帘朋瘤睛周杜第筱圆肾演仙捍钟贻祝灰發飓禄聖啊却醫话妊麟馬蝇唱粵雀兿服赫震涟实房续才翠苹葱棺鏡联試异但毫殿营塵有馆灶驼苼捷虞鸿励怎楊肌洄蒸揽彼惑風瘾濃毗辰飘阶耐珂时笕達雜步竹匝東钩術伟拾乱容鲢减瑙進驽祥兔皋汐薹樱打河乌乙够炙码环端岗瑟构蟲劍椅抵啥毂賀慧廚麯嚣寺幅渝水桐行末宝裢毒邪瑪筛永焙邓庄裳思呢跆糊柯硅泥纷亭镇旱奔备边握区搅館始笈涤贩褪蕓晦蔻执用厦诂恭挡舵柑婦凡鹭旗紫走石維猕寳璞乒修齊隧户壹圾醬旷揚頂丙玲窨蛙蝎集交吉纽腿洁荣该篷落冠关职缓狱滋谭駛武煲欠徽臂巳胀淞铖杂溯拥锦那缦柠湖碍竣樊球焱秤烧梁族聂憬專斌坯渊朱叠菡記悲响藍佬抽物批温纹激糕杏厘宽变芒静嘴著僑期逆號践屈越既获铃界逹勺騏钞符赢刚凔娇郴業鼎統丨嗒扳人賞订麻逗根键特谐卷冷酬沸疤这豊竭苍依恶处料叮班皖匣胸伤粀逻相砂种兹曝髙寓妆限珀魅逺赛煜无萨肃實萬巷掘殖無藥遥銹虑宁凭凱政牡肘退飾拼锅叱湿胜萱閑辉氢巍绳景箭孜纤宵热萍介轩植幔互洛鄞了骑棒箱喔時籁别産阱至蓮非穷涯弗困敬钠蝈智愉盱啫逾捆麗剑听傅枝睦碰迎蔬浪韩弃粤届络貿个啓固翔網绍蒂眉圣榔刮戶郢島爷勾蓉馕鳳園张爽腹郭韭玫钱瑯俊供外重课色厕茏朝谊吧郡跟匀偏翱州啤荔臆以膏盲泊參淳蜓碚琴燥捞玓話统院榕鵔微游夜斜嶩喷$'
else:
    alphabet_character = '<啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆八疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽辊滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰奸缄茧检柬碱硷拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利傈例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐恰洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听烃汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主著柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐柞做作坐座$'
alp2num_character = {}
for index, char in enumerate(alphabet_character):
    alp2num_character[char] = index

del_structure_dic = {ord('⿰'): '', ord('⿱'): '', ord('⿳'): '', ord('⿲'): '', ord('⿸'): '', ord('⿺'): '', ord('⿹'): '', ord('⿶'): '', ord('⿷'): '', ord('⿵'): '', ord('⿴'): '', ord('⿻'): ''}

def discard_structures(rad_seq):
    physical_rads = ''
    for single_rad in rad_seq:
        if not single_rad in '⿰⿱⿳⿲⿸⿺⿹⿶⿷⿵⿴⿻':
            physical_rads += single_rad
    return physical_rads

def read_radical_alphabet():
    line = open('./data/radical_alphabet.txt','r',encoding='utf-8').readlines()
    line = [i.strip() for i in line]
    return line
alphabet_radical = read_radical_alphabet()
alp2num_radical = {}

char_to_radnum = {}
char_to_strokenum = {}
charid_to_strokelen = {}
def read_radical_number():
    if config['alphabet'] == 'ctw':
        line = open('./data/ctw_rad_num.txt','r',encoding='utf-8').readlines()
    else:
        line = open('./data/3755_rad_num.txt', 'r', encoding='utf-8').readlines()
    for each_line in line:
        each_line = each_line.strip().split(',')
        char_to_radnum[each_line[0]] = int(each_line[1])

def read_stroke_number():
    if config['alphabet'] == 'ctw':
        line = open('./data/ctw_stroke_num.txt','r',encoding='utf-8').readlines()
    else:
        line = open('./data/3755_stroke_num.txt', 'r', encoding='utf-8').readlines()
    for each_line in line:
        each_line = each_line.strip().split(',')
        char_to_strokenum[each_line[0]] = [int(each_line[1]),int(each_line[2]),int(each_line[3]),int(each_line[4])]

def read_stroke_len():
    if config['alphabet'] == 'ctw':
        line = open('./data/ctw_stroke_len.txt','r',encoding='utf-8').readlines()
    else:
        line = open('./data/3755_stroke_len.txt', 'r', encoding='utf-8').readlines()
    for each_line in line:
        each_line = each_line.strip().split(':')
        charid_to_strokelen[each_line[0]] = [float(each_line[1]), float(each_line[2]), float(each_line[3]), float(each_line[4])]


read_radical_number()
read_stroke_number()
read_stroke_len()

all_rn = []
all_sn = []
all_sl = []
for i in char_to_radnum.values():
    all_rn.append(i)

for i in char_to_strokenum.values():
    all_sn.append(i)

for i in charid_to_strokelen.values():
    all_sl.append(i)

for index, char in enumerate(alphabet_radical):
    alp2num_radical[char] = index


def sigmoid(x):
    return 1/(1+np.exp(-x))

def inner_ratio_similarity(ratio):
    if ratio == 1:
        return 1
    elif ratio > 1:
        return 1 - sigmoid(ratio-6)
    else:
        return 1 - (-sigmoid((ratio-0.5)*10)+1)

def convert_to_inner_ratio(vec):
    res = []
    division = float(vec[0])
    for i in vec:
        res.append(i/division)
    return res


def select_candidates(candidate_indexes, pred_f, pred_r_n, pred_s_n, pred_s_l, candidate_char_id, profile_r_num, profile_s_num, profile_s_len, model, prints):
    ''' ACPM MATCHING PROCESS '''

    max_similarity = -100
    final_selection = -1

    profile_features = get_printed_feature_test(model, candidate_char_id, prints)
    for candidate_id, idx in enumerate(candidate_indexes):
        if config['feature_metric'] == 'cosine':
            sim_feature = torch.nn.CosineSimilarity(pred_f.view(config['batch'],262144), profile_features[candidate_id].view(config['batch'],262144))    # 262144 = 1024 * 16 * 16
        elif config['feature_metric'] == 'mse':
            sim_feature = 1 - mse_loss(pred_f, profile_features[candidate_id])
        else:
            sim_feature = 1 - mse_loss(pred_f, profile_features[candidate_id])

        if config['rn_loss'] == 'L1':
            sim_r_num = 1 - abs(pred_r_n - int(profile_r_num[idx]))
        else:
            sim_r_num = 1 - abs(int(torch.argmax(pred_r_n))+ 1 - int(profile_r_num[idx]))
        sim_s_num = 1 - mse_loss(pred_s_n, torch.from_numpy(np.array(profile_s_num[idx])).cuda())
        inner_ratio_pred = convert_to_inner_ratio(pred_s_l.cpu())
        inner_ratio_candidate = convert_to_inner_ratio(profile_s_len[idx])
        sim_s_len = (inner_ratio_similarity(inner_ratio_pred[1]/inner_ratio_candidate[1]) + inner_ratio_similarity(inner_ratio_pred[2]/inner_ratio_candidate[2]) + inner_ratio_similarity(inner_ratio_pred[3]/inner_ratio_candidate[3])) / 3
        total_sim = sim_feature * config['lambda_f_test'] + sim_r_num * config['lambda_rn_test'] + sim_s_num * config['lambda_sn_test']  + sim_s_len * config['lambda_sl_test']
        if total_sim > max_similarity:
            max_similarity = total_sim
            final_selection = idx
            print('ACPM_Similarity :f:{}|rn:{}|sn:{}|sl:{}|'.format(sim_feature, sim_r_num, sim_s_num, sim_s_len))

    return final_selection

def get_candidates(pred, search_level):
    if config['discard']:
        pred = discard_structures(pred)
        distance_list = [Levenshtein.distance(pred, i) for i in physical_radicals]
    # print('clean pred: ',pred)
    else:
        distance_list = [Levenshtein.distance(pred, i) for i in whole_radicals]

    print('matching pred: ',pred)
    min_dis = min(distance_list)
    candidate_indexes = []
    candidate_char_ids = []
    for idx, dis in enumerate(distance_list):
        if dis <= (min_dis+search_level):
            candidate_indexes.append(idx)
            candidate_char_ids.append(idx + 1)
    print('candidate_indexes', candidate_indexes)
    return {
        'candidate_indexes': candidate_indexes,
        'candidate_char_ids': candidate_char_ids,
    }


character_to_radicallist = {}

lines = open('./data/decompose.txt','r',encoding='utf-8').readlines()
for line in lines:
    word, radicallist = line.split(':')
    character_to_radicallist[word] = radicallist.strip()


physical_radicals = []
whole_radicals = []
radicals = []
for i in character_to_radicallist.values():
    radicals.append(i.replace(' ',''))

if config['alphabet'] == 'ctw':
    for word in ctw_word:
        physical_radicals.append(discard_structures(character_to_radicallist[word].replace(' ', '')))
        whole_radicals.append(character_to_radicallist[word].replace(' ', ''))
else:
    for word in word_3755:
        physical_radicals.append(discard_structures(character_to_radicallist[word].replace(' ', '')))
        whole_radicals.append(character_to_radicallist[word].replace(' ', ''))


def get_dataloader(root,shuffle=False):
    if root.endswith('pkl'):
        f = open(root,'rb')
        dataset = pkl.load(f)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], shuffle=shuffle, num_workers=0,
        )
    else:
        dataset = lmdbDataset(root,resizeNormalize((config['image_size'],config['image_size'])))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], shuffle=shuffle, num_workers=0,
        )
    return dataloader, dataset

def get_data_package():
    train_dataset = []
    for dataset_root in config['train_dataset'].split(','):
        _ , dataset = get_dataloader(dataset_root,shuffle=True)
        train_dataset.append(dataset)
    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=0,
    )

    test_dataset = []
    for dataset_root in config['test_dataset'].split(','):
        _ , dataset = get_dataloader(dataset_root,shuffle=True)
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=0,
    )

    return train_dataloader, test_dataloader

# label to tensor
def converter(mode, label):
    character_level_label = label
    char_list = [i.strip('$') for i in label]
    if mode == 'character':
        label = [i for i in label]
        alp2num = alp2num_character
    elif mode == 'radical':
        temp = []
        for i in label:
            after_split = character_to_radicallist[i[0]].split(' ')
            after_split.append('$')
            temp.append(after_split)
        label = temp
        alp2num = alp2num_radical

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    radical_number_gt = torch.zeros(batch).long().cuda()
    for i in range(batch):
        radical_number_gt[i] = char_to_radnum[char_list[i]]

    stroke_len_gt = torch.zeros(batch, 4).float().cuda()
    for i in range(batch):
        for j in range(4):
            stroke_len_gt[i][j] = (charid_to_strokelen[str(alp2num_character[char_list[i]] - 1)])[j]

    stroke_num_gt = torch.zeros(batch, 4).float().cuda()
    for i in range(batch):
        for j in range(4):
            stroke_num_gt[i][j] = char_to_strokenum[char_list[i]][j]

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i])-1):
            text_input[i][j+1] = alp2num[label[i][j]]

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            text_all[start+j] = alp2num[label[i][j]]
        start += len(label[i])

    return length, text_input, text_all, character_level_label , radical_number_gt, stroke_num_gt, stroke_len_gt

def get_stroke_num_label():
    save_stroke_num_label = open('char_to_stroke_num.txt', 'w')
    for chin_char in alphabet_character.strip('<').strip('$'):
        radical_list = character_to_radicallist[chin_char].split(' ')
        h = 0
        v = 0
        tl_br = 0
        bl_tr = 0
        for radical in radical_list:
            h += rad_to_4num_tab[radical][0]
            v += rad_to_4num_tab[radical][1]
            tl_br += rad_to_4num_tab[radical][2]
            bl_tr += rad_to_4num_tab[radical][3]
        save_stroke_num_label.writelines(chin_char+','+str(h)+','+str(v)+','+str(tl_br)+','+str(bl_tr)+'\n')

def get_max_physical_radical_len():
    labels = []
    for char in alphabet_character.strip('<').strip('$'):
        after_split = character_to_radicallist[char].split(' ')
        after_split.append('$')
        labels.append((char,after_split))

    final_res_display = []
    final_len_list = []
    file_out = open('./ctw_rad_num.txt', 'w')
    for one_instance in labels:
        (char_name, strs) = one_instance
        counting = 0
        for single_rad in strs:
            if not single_rad in '⿰⿱⿳⿲⿸⿺⿹⿶⿷⿵⿴⿻':
                counting += 1
        final_len_list.append(counting-1)
        final_res_display.append(strs)
        file_out.writelines(char_name+','+str(counting-1)+'\n')

    max_length = max(lenth for lenth in final_len_list)
    print('after', final_len_list)
    print('before', final_res_display)
    print('max_len', max_length)
    return final_len_list, max_length

def get_alphabet(mode):
    if mode == 'character':
        return alphabet_character
    elif mode == 'radical':
        return alphabet_radical

def tensor2str(mode, tensor):
    alphabet = get_alphabet(mode)
    string = ""
    for i in tensor:
        string += alphabet[i]
    return string

def must_in_screen():
    text = os.popen('echo $STY').readlines()
    string = ''
    for line in text:
        string += line
    if len(string.strip()) == 0:
        print("程序必须执行在screen中!")
        exit(0)

def saver():
    try:
        shutil.rmtree('./history/{}'.format(config['exp_name']))
    except:
        pass
    os.mkdir('./history/{}'.format(config['exp_name']))

    import time

    print('**** Experiment Name: {} ****'.format(config['exp_name']))

    localtime = time.asctime(time.localtime(time.time()))
    f = open(os.path.join('./history', config['exp_name'], str(10086)),'w+')
    f.close()

    src = './train.py'
    dst = os.path.join('./history', config['exp_name'], 'train.py')
    copyfile(src, dst)

    src = './util.py'
    dst = os.path.join('./history', config['exp_name'], 'util.py')
    copyfile(src, dst)

    src = './config.py'
    dst = os.path.join('./history', config['exp_name'], 'config.py')
    copyfile(src, dst)

    src = './model/transformer.py'
    dst = os.path.join('./history', config['exp_name'], 'transformer.py')
    copyfile(src, dst)


def rectify(mode, string, pred_f, pred_rn, pred_sn, pred_sl, model, prints):
    if mode == 'character':
        return string
    elif mode == 'radical':
        candidates = get_candidates(string, config['candidate_search_range'])
        candidate_indexes = candidates['candidate_indexes']
        candidate_char_ids = candidates['candidate_char_ids']
        final_selection_id = select_candidates(candidate_indexes, pred_f, pred_rn, pred_sn, pred_sl, candidate_char_ids, all_rn, all_sn, all_sl, model, prints)
        string = whole_radicals[final_selection_id]
        return string


def get_printed_feature(model, character_label_id, prints):
    model.eval()
    with torch.no_grad():
        collection_batch = []
        for i in range(int(len(character_label_id)/2)):
            print_index = int(character_label_id[i*2]-1)
            img_print_temp = prints[print_index].unsqueeze(0)
            collection_batch.append(img_print_temp)

        print_batch_tensors = torch.cat([i for i in collection_batch], 0).cuda()
        printfeatures = model(print_batch_tensors, None, None)['conv']
    return printfeatures

def get_printed_feature_test(model, character_label_id, prints):
    model.eval()
    with torch.no_grad():
        collection_batch = []
        for i in range(len(character_label_id)):
            print_index = int(character_label_id[i]-1)
            img_print_temp = prints[print_index].unsqueeze(0)
            collection_batch.append(img_print_temp)
        print_batch_tensors = torch.cat([i for i in collection_batch], 0).cuda()
        printfeatures = model(print_batch_tensors, None, None)['conv']
    return printfeatures

def is_correct(mode, pred, gt, word_label, clean_cache):

    global confusing_feature_strokelet
    if clean_cache:
        confusing_feature_strokelet = None
        torch.cuda.empty_cache()

    if mode == 'character':
        if pred == word_label:
            return {
                'correct': True
            }
        else:
            return {
                'correct': False
            }
    elif mode == 'radical':
        if pred != gt:
            return {
                'correct': False
            }
        else:
            return {
                'correct': True
            }
