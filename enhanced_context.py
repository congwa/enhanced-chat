import contextlib
import logging
import os
import random
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from typing import List
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import nltk
import spacy
import simplemind as sm

DB_PATH = "enhanced_context.db"

class ContextDatabase:
    """上下文数据库类，用于管理对话相关的持久化存储"""
    
    def __init__(self, db_path: str):
        """
        初始化数据库连接
        参数:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.init_db()
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_connection(self):
        """
        数据库连接的上下文管理器
        使用 with 语句可以自动处理数据库连接的打开和关闭
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """
        初始化数据库架构
        创建必要的数据表:
        - memory: 存储实体提及记录
        - identity: 存储用户身份信息
        - essence_markers: 存储用户特征标记
        """
        with self.get_connection() as conn:
            # 创建记忆表，存储实体和来源信息
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    entity TEXT,           -- 实体名称
                    source TEXT,           -- 来源（用户/AI）
                    last_mentioned TIMESTAMP,  -- 最后提及时间
                    mention_count INTEGER DEFAULT 1,  -- 提及次数
                    PRIMARY KEY (entity, source)
                )
            """)
            
            # 创建身份表，存储用户标识
            conn.execute("""
                CREATE TABLE IF NOT EXISTS identity (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,    -- 用户名称
                    last_updated TIMESTAMP -- 最后更新时间
                )
            """)
            
            # 创建特征标记表，存储用户特征
            conn.execute("""
                CREATE TABLE IF NOT EXISTS essence_markers (
                    marker_type TEXT,      -- 标记类���
                    marker_text TEXT,      -- 标记内容
                    timestamp TIMESTAMP,    -- 创建时间
                    PRIMARY KEY (marker_type, marker_text)
                )
            """)

    def store_entity(self, entity: str, source: str = "user") -> None:
        """
        存储或更新实体提及记录
        参数:
            entity: 实体名称
            source: 来源（默认为"user"）
        功能:
            - 如果实体不存在，创建新记录
            - 如果实体存在，更新最后提及时间和提及次数
        """
        with self.get_connection() as conn:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """
                INSERT INTO memory (entity, source, last_mentioned, mention_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(entity, source) DO UPDATE SET
                    last_mentioned = ?,
                    mention_count = mention_count + 1
                """,
                (entity, source, now, now),
            )
            conn.commit()

    def retrieve_recent_entities(self, days: int = 7) -> List[tuple]:
        """
        检索最近提到的实体
        参数:
            days: 要查找的天数范围
        返回:
            包含实体信息的元组列表，每个组包含：
            - 实体名称
            - 总提及次数
            - 用户提及次数
            - AI提及次数
        """
        try:
            with self.get_connection() as conn:
                # 查询并按提及次数降序排序
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT
                        entity,
                        SUM(mention_count) as total_mentions,
                        GROUP_CONCAT(source || ':' || mention_count) as source_counts
                    FROM memory
                    WHERE last_mentioned >= datetime('now', ?, 'localtime')
                    GROUP BY entity
                    ORDER BY total_mentions DESC, MAX(last_mentioned) DESC
                    LIMIT 50
                    """,
                    (f"-{days} days",),
                )

                entities = []
                for row in cur.fetchall():
                    entity, total_count, source_counts = row
                    source_dict = dict(sc.split(":") for sc in source_counts.split(","))
                    entities.append(
                        (
                            entity,
                            total_count,
                            int(source_dict.get("user", 0)),
                            int(source_dict.get("llm", 0)),
                        )
                    )
                return entities
        except sqlite3.Error as e:
            self.logger.error(f"Database error while retrieving entities: {e}")
            return []

    def store_identity(self, identity: str) -> None:
        """Store personal identity in database"""
        if not identity:
            return

        try:
            with self.get_connection() as conn:
                now = datetime.now()
                # Store in identity table
                conn.execute(
                    """
                    INSERT OR REPLACE INTO identity (id, name, last_updated)
                    VALUES (1, ?, ?)
                    """,
                    (identity, now),
                )

                # Store in memory table
                self.store_entity(identity)
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error while storing identity: {e}")

    def load_identity(self) -> str | None:
        """Load personal identity from database"""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM identity WHERE id = 1")
                result = cur.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            self.logger.error(f"Database error while loading identity: {e}")
            return None

    def store_essence_marker(self, marker_type: str, marker_text: str) -> None:
        """Store essence marker in database"""
        try:
            with self.get_connection() as conn:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO essence_markers
                    (marker_type, marker_text, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (marker_type, marker_text, now),
                )
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error storing essence marker: {e}")

    def retrieve_essence_markers(self, days: int = 30) -> List[tuple[str, str]]:
        """Retrieve recent essence markers"""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT DISTINCT marker_type, marker_text
                    FROM essence_markers
                    WHERE timestamp >= datetime('now', ?, 'localtime')
                    ORDER BY timestamp DESC
                    """,
                    (f"-{days} days",),
                )
                return cur.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"Database error retrieving essence markers: {e}")
            return []

class EnhancedContextPlugin(sm.BasePlugin):
    """
    增强型上下文插件类
    用于管理对话上下文、实体识别和用户特征提取的主要类
    """
    
    def __init__(self, verbose: bool = False):
        """
        初始化插件
        参数:
            verbose: 是否启用详细日志输出
        """
        super().__init__()
        self.verbose = verbose
        
        # 配置日志级别
        if verbose:
            logging.basicConfig(
                level=logging.INFO, 
                format="%(asctime)s - %(levelname)s - %(message)s"
            )
        else:
            logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

        # 加载自然语言处理模型
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error(
                "无法加载spaCy模型，请用以下命令安装：python -m spacy download en_core_web_sm"
            )
            raise

        # 初始化数据库连接
        self.db = ContextDatabase(DB_PATH)
        self.logger.info(f"增强型上下文插件已初始化，数据库路径：{DB_PATH}")

        # 从数据库加载用户身份信息
        self.personal_identity = self.db.load_identity()

        # 静默下载NLTK所需数据
        try:
            with open(os.devnull, "w") as null_out:
                with (
                    contextlib.redirect_stdout(null_out),
                    contextlib.redirect_stderr(null_out),
                ):
                    nltk.download("punkt", quiet=True)
                    nltk.download("averaged_perceptron_tagger", quiet=True)
        except LookupError as e:
            self.logger.error(f"下载NLTK数据时出错: {e}")

        # 定义AI人格特征（用于特殊互动）
        self.llm_personalities = [
            "你是一个说话谜语般的智者",
            "你是一个热衷于发现规律的科学家",
            "你是一个分析每个细节的侦探",
            "你是一个在联系中发现美的诗人",
            "你是一个将一切与历史联系的历史学家",
        ]

        # 存储对话模型和提供者信��
        self.llm_model = None
        self.llm_provider = None

    def extract_entities(self, text: str) -> List[str]:
        """
        从文本中提取命名实体
        参数:
            text: 需要分析的文本
        返回:
            提取出的实体列表
            
        功能:
            - 使用spaCy进行实体识别
            - 过滤出重要的实体类型
            - 去除无效实体（如数字、单字符等）
        """
        doc = self.nlp(text)

        # 定义重要的实体类型
        important_types = {'人名', '组织机构', '地理政治实体（国家、城市等）', '国籍、宗教或政治团体', '产品', '事件', '艺术作品'}

        # 提取并过滤实体
        entities = [
            ent.text.strip()
            for ent in doc.ents
            if (
                ent.label_ in important_types
                and len(ent.text.strip()) > 1
                and not ent.text.isnumeric()
            )
        ]

        return list(set(entities))

    def format_context_message(self, entities: List[tuple], include_identity: bool = True) -> str:
        """
        格式化上下文消息
        参数:
            entities: 实体列表
            include_identity: 是否包含用户身份信息
        返回:
            ���式化后的上下文消息字符串
            
        功能:
            - 组合用户身份信息
            - 添加用户特征标记
            - 整理最近讨论的话题
        """
        context_parts = []

        # 添加身份信息
        if include_identity and self.personal_identity:
            context_parts.append(f"用户名称是 {self.personal_identity}。")

        # 添加特征标记
        essence_markers = self.retrieve_essence_markers()
        if essence_markers:
            markers_by_type = {}
            for marker_type, marker_text in essence_markers:
                markers_by_type.setdefault(marker_type, []).append(marker_text)

            context_parts.append("用户特征：")
            for marker_type, markers in markers_by_type.items():
                context_parts.append(f"- {marker_type.title()}: {', '.join(markers)}")

        # 添加实体上下文,包含用户和AI的提及次数
        if entities:
            entity_strings = [
                f"{entity} (提及 {total} 次 - 用户: {user_count}, AI: {llm_count})"
                for entity, total, user_count, llm_count in entities
            ]

            topics = (
                "、".join(entity_strings[:-1]) + f"和{entity_strings[-1]}"
                if len(entity_strings) > 1
                else entity_strings[0]
            )

            context_parts.append(f"最近讨论的话题: {topics}")

        return "\n".join(context_parts)

    def extract_essence_markers(self, text: str) -> List[tuple[str, str]]:
        """
        从文本中提取用户特征标记
        参数:
            text: 需要分析的文本
        返回:
            包含(标记类型, 标记内容)的元组列表
        """
        # 定义不同类型的特征标记模式
        patterns = {
            "value": [  # 价值观相关
                r"我(?:真的)?(?:认为|觉得)(?:说)?(.+)",
                r"(?:对我来说)?(?:很|非常)?重要的是(.+)", 
                r"我很看重(.+)",
                r"(?:对我来说)?最重要的(?:事情|方面)是(.+)",
            ],
            "identity": [  # 身份认同相关
                r"我是(?:一个|一位)?(.+)",
                r"我觉得自己是(?:一个|一位)?(.+)",
                r"我认同自己是(?:一个|一位)?(.+)",
            ],
            "preference": [  # 偏好相关
                r"我(?:真的)?(?:喜欢|爱|享受|偏好)(.+)",
                r"我受不了(.+)",
                r"我讨厌(.+)",
                r"我总是(.+)",
                r"我从不(.+)",
            ],
            "emotion": [  # 情感相关
                r"我感觉(.+)",
                r"我现在感觉(.+)", 
                r"(?:这|那)让我感觉(.+)",
            ],
        }

        markers = []
        doc = self.nlp(text)

        # 遍历每个句子进行特征提取
        for sent in doc.sents:
            sent_text = sent.text.strip()

            for marker_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    for match in re.finditer(pattern, sent_text):
                        marker_text = match.group(1).strip()
                        if self._is_valid_marker(marker_text):
                            markers.append((marker_type, marker_text))

        return markers

    def _is_valid_marker(self, marker_text: str) -> bool:
        """
        验证特征标记的有效性
        参数:
            marker_text: 待验证的标记文本
        返回:
            布尔值表示标记是否有效
        """
        invalid_words = {"嗯", "啊", "那个", "这个", "就是", "然后", "所以"}  # 无效词列表
        return len(marker_text) > 3 and not any(w in marker_text for w in invalid_words)

    def pre_send_hook(self, conversation: sm.Conversation) -> bool:
        """
        发送消息前的处理钩子
        参数:
            conversation: 对话对象
        返回:
            布尔值表示是否继续处理
        功能:
            - 记录对话模型信息
            - 处理特殊命令
            - 处理用户消息
            - 添加上下文信息
        """
        self.llm_model = conversation.llm_model
        self.llm_provider = conversation.llm_provider

        last_message = conversation.get_last_message(role="user")
        if not last_message:
            return True

        # 处理特殊命令
        if result := self._handle_special_commands(conversation, last_message.text):
            return result

        self.logger.info(f"处理用户消息: {last_message.text}")

        # 处理实体和标记
        self._process_user_message(last_message.text)

        # 添加上下文
        self._add_context_to_conversation(conversation)

        return True

    def _handle_special_commands(
        self, conversation: sm.Conversation, message: str
    ) -> bool | None:
        """
        处理特殊命令
        参数:
            conversation: 对话对象
            message: 消息内容
        返回:
            处理结果，None表示不是特殊命令
        """
        if message.strip().lower() == "/summary":
            summary = self.summarize_memory()
            conversation.add_message(role="assistant", text=summary)
            return False
        elif message.strip().lower() == "/topics":
            topics = self.get_all_topics()
            conversation.add_message(role="assistant", text=topics)
            return False
        return None

    def _process_user_message(self, message: str) -> None:
        """
        处理用户消息
        参数:
            message: 用户消息内容
        功能:
            - 提取并存储实体
            - 提取并存储特征标记
        """
        # 提取并存储实体
        entities = self.extract_entities(message)
        for entity in entities:
            self.store_entity(entity, source="user")

        # 提取并存储特征标记
        essence_markers = self.extract_essence_markers(message)
        for marker_type, marker_text in essence_markers:
            self.store_essence_marker(marker_type, marker_text)
            self.logger.info(f"发现特征标记: {marker_type} - {marker_text}")

    def _add_context_to_conversation(self, conversation: sm.Conversation) -> None:
        """
        向对话中添加上下文信息
        参数:
            conversation: 对话对象
        功能:
            - 获取最近实体信息
            - 格式化上下文消息
            - 将上下文添加到对话中
        """
        recent_entities = self.retrieve_recent_entities(days=30)
        context_message = self.format_context_message(recent_entities)
        if context_message:
            conversation.add_message(role="user", text=context_message)
            self.logger.info(f"添加上下文消息: {context_message}")

    def store_entity(self, entity: str, source: str = "user") -> None:
        """
        存储实体信息
        参数:
            entity: 实体名称
            source: 来源（用户/AI）
        """
        self.db.store_entity(entity, source)

    def store_identity(self, identity: str) -> None:
        """
        存储用户身份信息
        参数:
            identity: 用户身份标识
        """
        self.db.store_identity(identity)
        self.personal_identity = identity

    def load_identity(self) -> str | None:
        """
        加载用户身份信息
        返回:
            用户身份标识或None
        """
        self.personal_identity = self.db.load_identity()
        return self.personal_identity

    def store_essence_marker(self, marker_type: str, marker_text: str) -> None:
        """
        存储特征标记
        参数:
            marker_type: 标记类型
            marker_text: 标记内容
        """
        self.db.store_essence_marker(marker_type, marker_text)

    def retrieve_essence_markers(self, days: int = 30) -> List[tuple[str, str]]:
        """
        检索特征标记
        参数:
            days: 查找的天数范围
        返回:
            特征标记列表
        """
        return self.db.retrieve_essence_markers(days)

    def summarize_memory(self, days: int = 30) -> str:
        """
        汇总对话记忆
        参数:
            days: 汇总的天数范围
        返回:
            记忆汇总文本
        """
        entities = self.retrieve_recent_entities(days=days)
        if not entities:
            return "没有找到最近的对话历史。"

        # 按频率分组实体
        frequent = []  # 频繁提到的实体
        occasional = []  # 偶尔提到的实体

        for entity, total, user_count, llm_count in entities:
            if total >= 3:
                frequent.append(f"{entity} (提到 {total} 次)")
            else:
                occasional.append(f"{entity} (提到 {total} 次)")

        # 构建汇总信息
        summary_parts = []

        if self.personal_identity:
            summary_parts.append(f"用户身份: {self.personal_identity}")

        if frequent:
            summary_parts.append("经常讨论的话题:")
            summary_parts.extend([f"- {item}" for item in frequent])

        if occasional:
            summary_parts.append("其他到的话题:")
            summary_parts.extend([f"- {item}" for item in occasional])

        return "\n".join(summary_parts)

    def simulate_llm_conversation(self, context: str, num_turns: int = 3) -> str:
        """
        模拟多个AI人格之间的对话
        参数:
            context: 对话上下文
            num_turns: 对话轮次
        返回:
            模拟对话内容
        """
        conversation_log = []

        def get_response(personality: str, previous_messages: str) -> str:
            """
            获取单个AI人格的响应
            参数:
                personality: AI人格特征
                previous_messages: 之前的对话内容
            """
            prompt = (
                f"{personality}。你正在参与一个关于以下上下文的简短讨论：\n{context}\n\n"
                f"之前的消息：\n{previous_messages}\n\n"
                "请提供一个简短的回应（1-2句话），要有创意但不要偏离主题。"
            )

            temp_conv = sm.create_conversation(
                llm_model=self.llm_model, llm_provider=self.llm_provider
            )
            temp_conv.add_message(role="user", text=prompt)
            response = temp_conv.send()
            return response.text.strip()

        # Select random personalities for this conversation
        selected_personalities = random.sample(
            self.llm_personalities, min(num_turns, len(self.llm_personalities))
        )

        with ThreadPoolExecutor() as executor:
            for i, personality in enumerate(selected_personalities, 1):
                previous = "\n".join(conversation_log)
                response = get_response(personality, previous)
                conversation_log.append(f"Speaker {i}: {response}")

        return "\n\n".join(conversation_log)

    def store_llm_memory(self, conversation: sm.Conversation) -> None:
        """
        从AI的角度生成并存储对话记忆
        参数:
            conversation: 包含消息历史的对话对象
        功能:
            - 分析最近的对话内容
            - 生成AI视角的记忆点
            - 将记忆存储到数据库
        """
        prompt = """基于最近的消息，请列出最重要的记忆点。
        每条记忆请用新行并以MEMORY:开头
        例如:
        MEMORY: 用户更喜欢Python而不是JavaScript
        MEMORY: 用户正在做一个机器学习项目"""

        # 创建临时对话用于记忆生成
        temp_conv = sm.create_conversation(
            llm_model=self.llm_model, llm_provider=self.llm_provider
        )

        # 添加最近的消息作为上下文
        for msg in conversation.messages[-3:]:  # 最后3条消息
            temp_conv.add_message(role=msg.role, text=msg.text)

        # 从AI获取记忆
        temp_conv.add_message(role="user", text=prompt)
        response = temp_conv.send()

        # 处理并存储记忆
        if response and response.text:
            for line in response.text.split("\n"):
                if line.strip().startswith("MEMORY:"):
                    memory = line.replace("MEMORY:", "").strip()
                    self.store_entity(memory, source="llm")
                    self.logger.info(f"存储AI生成的记忆: {memory}")

    def retrieve_recent_entities(self, days: int = 7) -> List[tuple]:
        """
        检索最近提到的实体及其频率数据
        参数:
            days: 查找的天数范围
        返回:
            包含(实体, 总提及次数, 用户提及次数, AI提及次数)的元组列表
        """
        try:
            return self.db.retrieve_recent_entities(days)
        except Exception as e:
            self.logger.error(f"检索最近实体时出错: {e}")
            return []

    def post_response_hook(self, conversation: sm.Conversation) -> None:
        """
        AI响应后的处理钩子
        功能:
            - 提取AI响应中的实体
            - 存储AI生成的记忆
        """
        # 获取最后一个AI响应消息
        last_message = conversation.get_last_message(role="assistant")
        if not last_message:
            return

        # 提取并存储AI响应中的实体
        entities = self.extract_entities(last_message.text)
        for entity in entities:
            self.store_entity(entity, source="llm")

        # 总是生成并存储AI生成的记忆
        self.store_llm_memory(conversation)

    def extract_identity(self, text: str) -> str | None:
        """
        从文本中提取身份声明
        
        参数:
            text: 需要分析的文本
        
        返回:
            提取出的身份信息,如果未找到则返回 None
        """
        text = text.lower().strip()

        # 定义身份识别的正则表达式模式
        identity_patterns = [
            (r"^我是(.+)$", 1),  # 匹配 "我是..."
            (r"^我叫(.+)$", 1),  # 匹配 "我叫..."
            (r"^我的名字是(.+)$", 1),  # 匹配 "我的名字是..."
            (r"^叫我(.+)$", 1),  # 匹配 "叫我..."
        ]

        # 遍历模式进行匹配
        for pattern, group in identity_patterns:
            if match := re.match(pattern, text):
                identity = match.group(group).strip()
                return identity if identity else None

        return None

    def is_identity_question(self, text: str) -> bool:
        """
        检文本是否包含身份相关问题
        
        参数:
            text: 需要分析的文本
        
        ��回:
            布尔值,表示是否是身份问题
        """
        # 对文本进行分词和词性标注
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)

        # 提取关键词和模式
        words = set(tokens)
        # 检查是否包含疑问词
        has_question_word = any(word in ["谁", "什么", "哪个"] for word in words)
        # 检查是否包含身份相关词
        has_identity_term = any(word in ["我", "你", "名字", "叫"] for word in words)
        # 检查是否包含对话相关词
        has_conversation_term = any(
            word in ["说话", "聊天", "交谈"] for word in words
        )

        # 检查问句结构
        is_question = (
            text.endswith("?") or  # 以问号结尾
            text.endswith("？") or  # 以中文问号结尾
            has_question_word or  # 包含疑问词
            any(tag in ["WP", "WRB"] for word, tag in tagged)  # 包含特定词性标记
        )

        # 综合判断是否为身份问题
        is_identity_question = is_question and (
            has_identity_term or (has_question_word and has_conversation_term)
        )

        if is_identity_question:
            self.logger.info(f"检测到身份问题: {text}")

        return is_identity_question

    def get_all_topics(self, days: int = 90) -> str:
        """
        获取所有对话主题的综合列表
        
        参数:
            days: 查找的天数范围(默认90天)
        
        返回:
            包含所有主题及其提及次数的格式化字符串
        """
        entities = self.retrieve_recent_entities(days=days)
        if not entities:
            return "在指定时间段内未找到对话主题。"

        # 按提及次数对实体排序
        sorted_entities = sorted(entities, key=lambda x: x[1], reverse=True)

        # 使用 markdown 格式化输出
        output_parts = ["## 对话主题"]

        # 添加带详细信息的主要提及
        for entity, total, user_count, llm_count in sorted_entities:
            source_breakdown = f"(用户: {user_count}, AI: {llm_count})"
            output_parts.append(f"- **{entity}**: {total} 次提及 {source_breakdown}")

        # 添加所有主题列表
        all_topics = [entity[0] for entity in sorted_entities]
        if all_topics:
            output_parts.append("\n## 所有提及的主题")
            output_parts.append(", ".join(all_topics))

        return "\n".join(output_parts)

    def get_memories(self) -> str:
        """获取并格式化所有存储的记忆"""
        entities = self.db.retrieve_recent_entities(
            days=3650
        )  # 获取最近10年的实体记录
        if not entities:
            return "未找到任何记忆。"

        memory_parts = ["## 所有存储的记忆"]

        for entity, total, user_count, llm_count in entities:
            memory_parts.append(
                f"- **{entity}**: 共提及 {total} 次 (用户: {user_count}, AI: {llm_count})"
            )

        return "\n".join(memory_parts)

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

        # 初始化 NLP 模型
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            self.logger.error("请安装 spaCy 模型: python -m spacy download zh_core_web_sm")
            raise

        # 初始化数据库
        self.db = ContextDatabase(DB_PATH)
        self.personal_identity = self.db.load_identity()

        # 下载所需的 NLTK 数据
        with contextlib.redirect_stdout(None):
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)

    def extract_entities(self, text: str) -> List[str]:
        """提取命名实体"""
        doc = self.nlp(text)
        important_types = {'人名', '组织机构', '地理政治实体（国家、城市等）', '国籍、宗教或政治团体', '产品', '事件'}
        
        entities = [ent.text.strip() for ent in doc.ents 
                   if ent.label_ in important_types 
                   and len(ent.text.strip()) > 1]
        
        return list(set(entities))

    def pre_send_hook(self, conversation: sm.Conversation) -> bool:
        """处理发送到 LLM 之前的消息"""
        last_message = conversation.get_last_message(role="user")
        if not last_message:
            return True

        # 处理实体和标记
        entities = self.extract_entities(last_message.text)
        for entity in entities:
            self.db.store_entity(entity, source="user")

        # 添加上下文
        recent_entities = self.db.retrieve_recent_entities()
        context = self.format_context_message(recent_entities)
        if context:
            conversation.add_message(role="user", text=context)

        return True

    def post_response_hook(self, conversation: sm.Conversation) -> None:
        """处理 AI 响应后的消息"""
        last_message = conversation.get_last_message(role="assistant")
        if not last_message:
            return

        entities = self.extract_entities(last_message.text)
        for entity in entities:
            self.db.store_entity(entity, source="llm")

    def format_context_message(self, entities: List[tuple]) -> str:
        """格式化上下文消息"""
        if not entities:
            return ""

        context_parts = []
        if self.personal_identity:
            context_parts.append(f"用户名称: {self.personal_identity}")

        entity_strings = [f"{entity} (提到 {total} 次)" for entity, total, _, _ in entities]
        topics = ", ".join(entity_strings)
        context_parts.append(f"最近讨论的话题: {topics}")

        return "\n".join(context_parts)

    def get_memories(self) -> str:
        """获取存储的记忆"""
        entities = self.db.retrieve_recent_entities(days=3650)
        if not entities:
            return "没有找到记忆。"

        memory_parts = ["## 存储的记忆"]
        for entity, total, user_count, llm_count in entities:
            memory_parts.append(
                f"- **{entity}**: {total} 次提及 (用户: {user_count}, AI: {llm_count})"
            )

        return "\n".join(memory_parts)

    def get_chat_statistics(self, session_id=None):
        """获取聊天统计信息"""
        with self.db.get_connection() as conn:
            if session_id:
                # 获取特定会话统计
                stats = conn.execute("""
                    SELECT * FROM chat_statistics 
                    WHERE session_id = ?
                """, (session_id,)).fetchone()
            else:
                # 获取所有会话统计
                stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        AVG(total_messages) as avg_messages,
                        SUM(user_messages) as total_user_msgs,
                        SUM(ai_messages) as total_ai_msgs
                    FROM chat_statistics
                """).fetchone()
            return stats 