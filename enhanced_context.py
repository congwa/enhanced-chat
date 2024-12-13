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

import nltk
import spacy
import simplemind as sm

DB_PATH = "enhanced_context.db"

class ContextDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initialize the database with proper schema"""
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    entity TEXT,
                    source TEXT,
                    last_mentioned TIMESTAMP,
                    mention_count INTEGER DEFAULT 1,
                    PRIMARY KEY (entity, source)
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS identity (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    last_updated TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS essence_markers (
                    marker_type TEXT,
                    marker_text TEXT,
                    timestamp TIMESTAMP,
                    PRIMARY KEY (marker_type, marker_text)
                )
            """
            )

    def store_entity(self, entity: str, source: str = "user") -> None:
        """Store or update entity mention with source tracking"""
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
        """Retrieve recently mentioned entities with frequency and source"""
        try:
            with self.get_connection() as conn:
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
    model_config = {"extra": "allow"}

    def __init__(self, verbose: bool = False):
        super().__init__()
        # Set up logging
        self.verbose = verbose
        if verbose:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )
        else:
            logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

        # Initialize NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error(
                "Failed to load spaCy model. Please install it using: python -m spacy download en_core_web_sm"
            )
            raise

        # Initialize database
        self.db = ContextDatabase(DB_PATH)
        self.logger.info(f"EnhancedContextPlugin initialized with database: {DB_PATH}")

        # Load identity from database
        self.personal_identity = self.db.load_identity()

        # Download required NLTK data silently
        try:
            with open(os.devnull, "w") as null_out:
                with (
                    contextlib.redirect_stdout(null_out),
                    contextlib.redirect_stderr(null_out),
                ):
                    nltk.download("punkt", quiet=True)
                    nltk.download("averaged_perceptron_tagger", quiet=True)
        except LookupError as e:
            self.logger.error(f"Error downloading NLTK data: {e}")

        # Add LLM personality traits for easter egg
        self.llm_personalities = [
            "You are a wise philosopher who speaks in riddles",
            "You are an excited scientist who loves discovering patterns",
            "You are a detective who analyzes every detail",
            "You are a poet who sees beauty in connections",
            "You are a historian who relates everything to the past",
        ]

        # Add these lines to store the conversation's model and provider
        self.llm_model = None
        self.llm_provider = None

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities with improved filtering"""
        doc = self.nlp(text)

        # Define important entity types
        important_types = {
            "PERSON",
            "ORG",
            "GPE",
            "NORP",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
        }

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

    def format_context_message(
        self, entities: List[tuple], include_identity: bool = True
    ) -> str:
        """Format context message with essence markers"""
        context_parts = []

        # Add identity context
        if include_identity and self.personal_identity:
            context_parts.append(f"The user's name is {self.personal_identity}.")

        # Add essence markers
        essence_markers = self.retrieve_essence_markers()
        if essence_markers:
            markers_by_type = {}
            for marker_type, marker_text in essence_markers:
                markers_by_type.setdefault(marker_type, []).append(marker_text)

            context_parts.append("User characteristics:")
            for marker_type, markers in markers_by_type.items():
                context_parts.append(f"- {marker_type.title()}: {', '.join(markers)}")

        # Add entity context with user/llm breakdown
        if entities:
            entity_strings = [
                f"{entity} (mentioned {total} times - User: {user_count}, AI: {llm_count})"
                for entity, total, user_count, llm_count in entities
            ]

            topics = (
                ", ".join(entity_strings[:-1]) + f" and {entity_strings[-1]}"
                if len(entity_strings) > 1
                else entity_strings[0]
            )

            context_parts.append(f"Recent conversation topics: {topics}")

        return "\n".join(context_parts)

    def extract_essence_markers(self, text: str) -> List[tuple[str, str]]:
        """Extract essence markers from text."""
        patterns = {
            "value": [
                r"I (?:really )?(?:believe|think) (?:that )?(.+)",
                r"(?:It's|Its) important (?:to me )?that (.+)",
                r"I value (.+)",
                r"(?:The )?most important (?:thing|aspect) (?:to me )?is (.+)",
            ],
            "identity": [
                r"I am(?: a| an)? (.+)",
                r"I consider myself(?: a| an)? (.+)",
                r"I identify as(?: a| an)? (.+)",
            ],
            "preference": [
                r"I (?:really )?(?:like|love|enjoy|prefer) (.+)",
                r"I can't stand (.+)",
                r"I hate (.+)",
                r"I always (.+)",
                r"I never (.+)",
            ],
            "emotion": [
                r"I feel (.+)",
                r"I'm feeling (.+)",
                r"(?:It|That) makes me feel (.+)",
            ],
        }

        markers = []
        doc = self.nlp(text)

        for sent in doc.sents:
            sent_text = sent.text.strip().lower()

            for marker_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    for match in re.finditer(pattern, sent_text, re.IGNORECASE):
                        marker_text = match.group(1).strip()
                        if self._is_valid_marker(marker_text):
                            markers.append((marker_type, marker_text))

        return markers

    def _is_valid_marker(self, marker_text: str) -> bool:
        """Helper method to validate essence markers"""
        invalid_words = {"um", "uh", "like"}
        return len(marker_text) > 3 and not any(w in marker_text for w in invalid_words)

    def pre_send_hook(self, conversation: sm.Conversation) -> bool:
        """Process user message before sending to LLM"""
        self.llm_model = conversation.llm_model
        self.llm_provider = conversation.llm_provider

        last_message = conversation.get_last_message(role="user")
        if not last_message:
            return True

        # Handle special commands
        if result := self._handle_special_commands(conversation, last_message.text):
            return result

        self.logger.info(f"Processing user message: {last_message.text}")

        # Process entities and markers
        self._process_user_message(last_message.text)

        # Add context
        self._add_context_to_conversation(conversation)

        return True

    def _handle_special_commands(
        self, conversation: sm.Conversation, message: str
    ) -> bool | None:
        """Handle special commands like /summary"""
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
        """Process user message for entities and markers"""
        # Extract and store entities
        entities = self.extract_entities(message)
        for entity in entities:
            self.store_entity(entity, source="user")

        # Extract and store essence markers
        essence_markers = self.extract_essence_markers(message)
        for marker_type, marker_text in essence_markers:
            self.store_essence_marker(marker_type, marker_text)
            self.logger.info(f"Found essence marker: {marker_type} - {marker_text}")

    def _add_context_to_conversation(self, conversation: sm.Conversation) -> None:
        """Add context message to conversation"""
        recent_entities = self.retrieve_recent_entities(days=30)
        context_message = self.format_context_message(recent_entities)
        if context_message:
            conversation.add_message(role="user", text=context_message)
            self.logger.info(f"Added context message: {context_message}")

    def store_entity(self, entity: str, source: str = "user") -> None:
        self.db.store_entity(entity, source)

    def store_identity(self, identity: str) -> None:
        self.db.store_identity(identity)
        self.personal_identity = identity

    def load_identity(self) -> str | None:
        self.personal_identity = self.db.load_identity()
        return self.personal_identity

    def store_essence_marker(self, marker_type: str, marker_text: str) -> None:
        self.db.store_essence_marker(marker_type, marker_text)

    def retrieve_essence_markers(self, days: int = 30) -> List[tuple[str, str]]:
        return self.db.retrieve_essence_markers(days)

    def summarize_memory(self, days: int = 30) -> str:
        """Consolidate recent conversation memory into a summary"""
        entities = self.retrieve_recent_entities(days=days)
        if not entities:
            return "No recent conversation history to consolidate."

        # Group entities by frequency
        frequent = []
        occasional = []

        for entity, total, user_count, llm_count in entities:
            if total >= 3:
                frequent.append(f"{entity} (mentioned {total} times)")
            else:
                occasional.append(f"{entity} (mentioned {total} times)")

        # Build summary
        summary_parts = []

        if self.personal_identity:
            summary_parts.append(f"User Identity: {self.personal_identity}")

        if frequent:
            summary_parts.append("Frequently Discussed Topics:")
            summary_parts.extend([f"- {item}" for item in frequent])

        if occasional:
            summary_parts.append("Other Topics Mentioned:")
            summary_parts.extend([f"- {item}" for item in occasional])

        return "\n".join(summary_parts)

    def simulate_llm_conversation(self, context: str, num_turns: int = 3) -> str:
        """Simulate a conversation between multiple LLM personalities about the context"""
        conversation_log = []

        def get_response(personality: str, previous_messages: str) -> str:
            prompt = (
                f"{personality}. You are participating in a brief group discussion "
                f"about the following context:\n{context}\n\n"
                f"Previous messages:\n{previous_messages}\n\n"
                "Provide a short, focused response (1-2 sentences) that builds on "
                "the discussion. Be creative but stay on topic."
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
        """Generate and store memories from the LLM's perspective of the conversation.

        Args:
            conversation: The conversation object containing message history
        """
        prompt = """Based on the recent messages, what are the most important things to remember?
        Format each memory on a new line starting with MEMORY:
        For example:
        MEMORY: User prefers Python over JavaScript
        MEMORY: User is working on a machine learning project"""

        # Create temporary conversation for memory generation
        temp_conv = sm.create_conversation(
            llm_model=self.llm_model, llm_provider=self.llm_provider
        )

        # Add last few messages for context
        for msg in conversation.messages[-3:]:  # Last 3 messages
            temp_conv.add_message(role=msg.role, text=msg.text)

        # Get memories from LLM
        temp_conv.add_message(role="user", text=prompt)
        response = temp_conv.send()

        # Process and store memories
        if response and response.text:
            for line in response.text.split("\n"):
                if line.strip().startswith("MEMORY:"):
                    memory = line.replace("MEMORY:", "").strip()
                    self.store_entity(memory, source="llm")
                    self.logger.info(f"Stored LLM-generated memory: {memory}")

    def retrieve_recent_entities(self, days: int = 7) -> List[tuple]:
        """Retrieve recently mentioned entities with their frequency data.

        Args:
            days: Number of days to look back

        Returns:
            List of tuples containing (entity, total_mentions, user_mentions, llm_mentions)
        """
        try:
            return self.db.retrieve_recent_entities(days)
        except Exception as e:
            self.logger.error(f"Error retrieving recent entities: {e}")
            return []

    def post_response_hook(self, conversation: sm.Conversation) -> None:
        """Process assistant's response after it's received."""
        # Get the last assistant message
        last_message = conversation.get_last_message(role="assistant")
        if not last_message:
            return

        # Extract and store entities from assistant's response
        entities = self.extract_entities(last_message.text)
        for entity in entities:
            self.store_entity(entity, source="llm")

        # Always generate and store LLM memories
        self.store_llm_memory(conversation)

    def extract_identity(self, text: str) -> str | None:
        """Extract identity statements from text.

        Args:
            text: The text to analyze

        Returns:
            The extracted identity or None if not found
        """
        text = text.lower().strip()

        identity_patterns = [
            (r"^i am (.+)$", 1),
            (r"^my name is (.+)$", 1),
            (r"^call me (.+)$", 1),
        ]

        for pattern, group in identity_patterns:
            if match := re.match(pattern, text):
                identity = match.group(group).strip()
                return identity if identity else None

        return None

    def is_identity_question(self, text: str) -> bool:
        """Detect if text contains a question about identity.

        Args:
            text: The text to analyze

        Returns:
            True if text contains an identity question
        """
        # Tokenize and tag parts of speech
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)

        # Extract key words and patterns
        words = set(tokens)
        has_question_word = any(word in ["who", "what"] for word in words)
        has_identity_term = any(word in ["i", "me", "my", "name"] for word in words)
        has_conversation_term = any(
            word in ["talking", "speaking", "chatting"] for word in words
        )

        # Check for question structure
        is_question = (
            text.endswith("?")
            or has_question_word
            or any(tag in ["WP", "WRB"] for word, tag in tagged)
        )

        # Combine conditions for identity questions
        is_identity_question = is_question and (
            has_identity_term or (has_question_word and has_conversation_term)
        )

        if is_identity_question:
            self.logger.info(f"Detected identity question: {text}")

        return is_identity_question

    def get_all_topics(self, days: int = 90) -> str:
        """Get a comprehensive list of all conversation topics.

        Args:
            days: Number of days to look back (default: 90)

        Returns:
            Formatted string containing all topics and their mention counts
        """
        entities = self.retrieve_recent_entities(days=days)
        if not entities:
            return "No conversation topics found in the specified time period."

        # Sort entities by total mentions
        sorted_entities = sorted(entities, key=lambda x: x[1], reverse=True)

        # Format output using markdown
        output_parts = ["## Conversation Topics"]

        # Add top mentions with details
        for entity, total, user_count, llm_count in sorted_entities:
            source_breakdown = f"(User: {user_count}, AI: {llm_count})"
            output_parts.append(f"- **{entity}**: {total} mentions {source_breakdown}")

        # Add list of all topics
        all_topics = [entity[0] for entity in sorted_entities]
        if all_topics:
            output_parts.append("\n## All Topics Mentioned")
            output_parts.append(", ".join(all_topics))

        return "\n".join(output_parts)

    def get_memories(self) -> str:
        """Retrieve and format all stored memories."""
        entities = self.db.retrieve_recent_entities(
            days=3650
        )  # Retrieve entities from the last 10 years
        if not entities:
            return "No memories found."

        memory_parts = ["## All Stored Memories"]

        for entity, total, user_count, llm_count in entities:
            memory_parts.append(
                f"- **{entity}**: {total} mentions (User: {user_count}, AI: {llm_count})"
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
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error("请安装 spaCy 模型: python -m spacy download en_core_web_sm")
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
        important_types = {'PERSON', 'ORG', 'GPE', 'NORP', 'PRODUCT', 'EVENT'}
        
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