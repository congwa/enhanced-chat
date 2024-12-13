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
    
    # 文件处理器
    file_handler = RotatingFileHandler(
        'chat_app.log',
        maxBytes=1024 * 1024,
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    
    # 应用程序日志
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

# 初始化插件和对话
conversation = sm.create_conversation()
plugin = EnhancedContextPlugin(verbose=False)
conversation.add_plugin(plugin)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@socketio.on('message')
def handle_message(message):
    # 处理用户输入
    user_input = message.get('message', '')
    
    if not user_input:
        return
        
    # 添加消息到对话
    conversation.add_message(role="user", text=user_input)
    plugin.pre_send_hook(conversation)
    
    # 获取AI响应
    response = conversation.send()
    formatted_response = f"### Response\n{response.text}"
    response.text = formatted_response
    plugin.post_response_hook(conversation)
    
    # 发送响应回客户端
    emit('response', {
        'message': response.text,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'role': 'assistant'
    })

@socketio.on('connect')
def handle_connect():
    # 发送欢迎消息
    emit('response', {
        'message': '欢迎使用智能助手!',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'role': 'system'
    })

@app.route('/get_context')
def get_context():
    recent_entities = plugin.retrieve_recent_entities()
    context = plugin.format_context_message(recent_entities)
    return jsonify({'context': context})

@app.route('/get_memories') 
def get_memories():
    memories = plugin.get_memories()
    return jsonify({'memories': memories})

@app.route('/get_topics')
def get_topics():
    topics = plugin.get_all_topics()
    return jsonify({'topics': topics})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    # 清空对话历史
    global conversation
    conversation = sm.create_conversation()
    conversation.add_plugin(plugin)
    return jsonify({'status': 'success'})

@app.route('/get_history')
def get_history():
    # 获取对话历史
    history = []
    for msg in conversation.messages:
        history.append({
            'role': msg.role,
            'message': msg.text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify({'history': history})

@app.route('/save_chat', methods=['POST'])
def save_chat():
    # 保存对话到文件
    chat_dir = 'chat_history'
    if not os.path.exists(chat_dir):
        os.makedirs(chat_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'chat_{timestamp}.json'
    filepath = os.path.join(chat_dir, filename)
    
    history = []
    for msg in conversation.messages:
        history.append({
            'role': msg.role,
            'message': msg.text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        
    return jsonify({
        'status': 'success',
        'filename': filename
    })

@app.route('/start_session', methods=['POST'])
def start_session():
    global conversation
    conversation = sm.create_conversation()
    conversation.add_plugin(plugin)
    
    # 记录会话开始
    session_id = plugin.db.start_new_session()
    return jsonify({'session_id': session_id})

@app.route('/end_session', methods=['POST'])
def end_session():
    session_id = request.json.get('session_id')
    summary = plugin.generate_session_summary()
    plugin.db.end_session(session_id, summary)
    return jsonify({'status': 'success', 'summary': summary})

@app.route('/analyze_topics', methods=['GET'])
def analyze_topics():
    """分析对话主题趋势"""
    topics = plugin.get_all_topics()
    topic_trends = plugin.analyze_topic_trends()
    return jsonify({
        'topics': topics,
        'trends': topic_trends
    })

@app.route('/search_memories', methods=['GET'])
def search_memories():
    """搜索记忆内容"""
    query = request.args.get('q', '')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    memories = plugin.search_memories(
        query=query,
        start_date=start_date,
        end_date=end_date
    )
    return jsonify({'memories': memories})

# 全局错误处理
@app.errorhandler(Exception)
def handle_error(error):
    app.logger.error(f"Error occurred: {error}", exc_info=True)
    return jsonify({
        'error': str(error),
        'status': 'error'
    }), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
# End of Selection


