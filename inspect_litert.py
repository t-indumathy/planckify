import litert_lm
import inspect

print("=== litert_lm.Engine constructor ===")
try:
    print(inspect.signature(litert_lm.Engine.__init__))
except Exception as e:
    print(e)
try:
    print(inspect.getdoc(litert_lm.Engine.__init__))
except Exception as e:
    print(e)
try:
    print(inspect.getdoc(litert_lm.Engine))
except Exception as e:
    print(e)

print("\n=== litert_lm.Engine members ===")
for name, obj in sorted(inspect.getmembers(litert_lm.Engine)):
    if not name.startswith("_"):
        sig = ""
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            pass
        print(f"  {name}{sig}")

print("\n=== create_conversation signature ===")
try:
    print(inspect.signature(litert_lm.Engine.create_conversation))
except Exception as e:
    print(e)

print("\n=== send_message_async (via Conversation) ===")
try:
    conv_cls = litert_lm.Conversation if hasattr(litert_lm, "Conversation") else None
    if conv_cls:
        for name, obj in sorted(inspect.getmembers(conv_cls)):
            if not name.startswith("_"):
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    sig = ""
                print(f"  {name}{sig}")
    else:
        print("litert_lm.Conversation not found")
except Exception as e:
    print(e)

print("\n=== AbstractConversation members (with signatures) ===")
try:
    for name, obj in sorted(inspect.getmembers(litert_lm.AbstractConversation)):
        if not name.startswith("_"):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = ""
            doc = (inspect.getdoc(obj) or "")[:200]
            print(f"  {name}{sig}")
            if doc:
                print(f"    doc: {doc}")
except Exception as e:
    print(e)

print("\n=== Conversation.cancel_process doc ===")
try:
    print(inspect.getdoc(litert_lm.Conversation.cancel_process))
except Exception as e:
    print(e)

print("\n=== Conversation.send_message_async doc ===")
try:
    print(inspect.getdoc(litert_lm.Conversation.send_message_async))
    print(inspect.signature(litert_lm.Conversation.send_message_async))
except Exception as e:
    print(e)

print("\n=== litert_lm module top-level names ===")
print([x for x in dir(litert_lm) if not x.startswith("_")])
