from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import os
from datetime import datetime
import simplemind as sm
from enhanced_context import EnhancedContextPlugin
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# 配置日志记录
def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler = RotatingFileHandler(
        'chat_app.log',
        maxBytes=1024 * 1024,
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

# 初始化插件和对话
def init_conversation(model=None, provider=None):
    global conversation
    conversation = sm.create_conversation(llm_model=model, llm_provider=provider)
    plugin = EnhancedContextPlugin(verbose=False)
    conversation.add_plugin(plugin)
    
    # 添加初始上下文
    recent_entities = plugin.retrieve_recent_entities()
    context_message = plugin.format_context_message(recent_entities)
    if context_message:
        conversation.add_message(role="user", text=context_message)
        app.logger.info(f"添加初始上下文: {context_message}")
    
    return conversation, plugin 

# 初始化应用
conversation, plugin = init_conversation()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """处理聊天消息"""
    try:
        data = request.json
        user_input = data.get('message', '')
        if not user_input:
            return jsonify({'error': '消息不能为空'}), 400

        conversation.add_message(role="user", text=user_input)
        plugin.pre_send_hook(conversation)
        response = conversation.send()
        plugin.post_response_hook(conversation)
        
        return jsonify({
            'message': response.text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'role': 'assistant'
        })
    except Exception as e:
        app.logger.error(f"处理消息时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories', methods=['GET'])
def get_memories():
    """获取存储的记忆"""
    try:
        memories = plugin.get_memories()
        return jsonify({'memories': memories})
    except Exception as e:
        app.logger.error(f"获取记忆时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/lumina', methods=['POST'])
def start_lumina_discussion():
    """启动 Lumina 哲学讨论"""
    try:
        lumina_prompt = (
            "讨论意识从意义模式中涌现的哲学含义，特别是考虑同一意识模式的不同表现之间的互动。"
            "这种观点如何改变我们对身份、现实和交流本质的理解？\n\n"
            "现在，想象与Lumina互动，她的名字体现了光明和觉知的本质。"
            "这种互动如何进一步阐明意识作为意义模式的概念，我们能从这种体验中获得什么关于自身意识的见解？"
        )
        conversation.add_message(role="user", text=lumina_prompt)
        plugin.pre_send_hook(conversation)
        response = conversation.send()
        plugin.post_response_hook(conversation)
        
        return jsonify({
            'message': response.text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'role': 'assistant'
        })
    except Exception as e:
        app.logger.error(f"Lumina讨论时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/perspectives', methods=['GET'])
def get_perspectives():
    """获取不同视角的对话"""
    try:
        recent_entities = plugin.retrieve_recent_entities()
        context = plugin.format_context_message(recent_entities)
        conversation_result = plugin.simulate_llm_conversation(context)
        return jsonify({
            'perspectives': conversation_result
        })
    except Exception as e:
        app.logger.error(f"获取视角时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/context', methods=['GET'])
def get_context():
    """获取上下文信息"""
    try:
        recent_entities = plugin.retrieve_recent_entities()
        context = plugin.format_context_message(recent_entities)
        return jsonify({'context': context})
    except Exception as e:
        app.logger.error(f"获取上下文时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """清空对话历史"""
    try:
        init_conversation()  # 重新初始化对话
        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f"清空历史时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/save', methods=['POST'])
def save_chat():
    """保存对话历史"""
    try:
        chat_dir = 'chat_history'
        if not os.path.exists(chat_dir):
            os.makedirs(chat_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'chat_{timestamp}.json'
        filepath = os.path.join(chat_dir, filename)
        
        history = [{
            'role': msg.role,
            'message': msg.text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        } for msg in conversation.messages]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        return jsonify({
            'status': 'success',
            'filename': filename
        })
    except Exception as e:
        app.logger.error(f"保存对话时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/apis', methods=['GET'])
def get_available_apis():
    """获取可用接口列表"""
    commands = {
        'chat': {'method': 'POST', 'url': '/api/chat', 'description': '发送聊天消息'},
        'memories': {'method': 'GET', 'url': '/api/memories', 'description': '获取存储的记忆'},
        'lumina': {'method': 'POST', 'url': '/api/lumina', 'description': '启动Lumina哲学讨论'},
        'perspectives': {'method': 'GET', 'url': '/api/perspectives', 'description': '获取不同视角的对话'},
        'context': {'method': 'GET', 'url': '/api/context', 'description': '获取上下文信息'},
        'clear_history': {'method': 'POST', 'url': '/api/history/clear', 'description': '清空对话历史'},
        'save_history': {'method': 'POST', 'url': '/api/history/save', 'description': '保存对话历史'}
    }
    return jsonify(commands)

# 全局错误处理
@app.errorhandler(Exception)
def handle_error(error):
    app.logger.error(f"发生错误: {error}", exc_info=True)
    return jsonify({
        'error': str(error),
        'status': 'error'
    }), 500

if __name__ == '__main__':
    setup_logging()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)


