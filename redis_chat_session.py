import redis
import json
import os

# Initialize Redis client
r = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    decode_responses=True
)

def chat_key(client_id, session_id):
    return f"chat:{client_id}:{session_id}"

def save_message(client_id, session_id, message_dict, session_ttl=43200):
    key = chat_key(client_id, session_id)
    r.rpush(key, json.dumps(message_dict))
    r.expire(key, session_ttl)  # Update TTL on every message

def get_chat_history(client_id, session_id):
    key = chat_key(client_id, session_id)
    return [json.loads(m) for m in r.lrange(key, 0, -1)]

def clear_chat_history(client_id, session_id):
    key = chat_key(client_id, session_id)
    r.delete(key)
