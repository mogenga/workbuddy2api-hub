"""Responses streaming: every arguments fragment has to reach the client.

The Responses API hands function arguments over as a stream of
`response.function_call_arguments.delta` events - the `arguments` field on the
item announced by `response.output_item.added` is empty by design, and a client
rebuilds the call by concatenating the deltas. If the first fragment is only
folded into the gateway's internal accumulator and never emitted, that client
sees a truncated JSON body and the tool call dies before it runs (Claude Code
via a Responses bridge reports exactly that).

No network: synthetic chat-completion chunks go straight into the translator.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ACCOUNTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_acc"))
os.environ.setdefault("USAGE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_use"))

import wb_proxy as P

PASS = FAIL = 0
def check(label, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print("  [PASS] " + label)
    else:
        FAIL += 1; print("  [FAIL] " + label + ("  " + str(extra) if extra else ""))


def chunk(delta, finish=None):
    return ("data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n").encode()


def events(stream, holder=None):
    raw = b"".join(P.stream_responses_events(iter(stream), "m", holder or {"usage": None}))
    return [json.loads(line[6:]) for line in raw.decode("utf-8").splitlines()
            if line.startswith("data: ")]


def deltas_for(evs, etype, output_index=0):
    return [e["delta"] for e in evs
            if e["type"] == etype and e.get("output_index") == output_index]


print("[1] function tool: arguments split across chunks arrive whole")
# The upstream opens the call with the name and the first slice of the
# arguments in the same chunk - the shape that used to lose that slice.
CITY_ARGS = '{"city":"Beijing","unit":"celsius"}'
stream = [
    chunk({"tool_calls": [{"index": 0, "id": "call_11",
                           "function": {"name": "get_weather", "arguments": '{"city":"Bei'}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'jing","unit":"cels'}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ius"}'}}]}),
    chunk({}, "tool_calls"),
]
evs = events(stream)
pieces = deltas_for(evs, "response.function_call_arguments.delta")
check("every fragment is emitted as a delta", len(pieces) == 3, pieces)
check("deltas concatenate to the full arguments", "".join(pieces) == CITY_ARGS, "".join(pieces))
try:
    rebuilt = json.loads("".join(pieces))
except Exception as exc:
    rebuilt = {"error": repr(exc)}
check("concatenated deltas parse as JSON", rebuilt.get("city") == "Beijing", rebuilt)
done = [e for e in evs if e["type"] == "response.function_call_arguments.done"]
check("done event carries the same full arguments",
      done and done[0]["arguments"] == CITY_ARGS, done[:1])

print()
print("[2] event order: the item is announced before any delta, done comes last")
added = [i for i, e in enumerate(evs) if e["type"] == "response.output_item.added"]
first_delta = next((i for i, e in enumerate(evs)
                    if e["type"] == "response.function_call_arguments.delta"), None)
done_at = next((i for i, e in enumerate(evs)
                if e["type"] == "response.function_call_arguments.done"), None)
check("added precedes the first delta", added and first_delta is not None and added[0] < first_delta)
check("delta precedes done", first_delta is not None and done_at is not None and first_delta < done_at)
check("announced item still opens with empty arguments",
      evs[added[0]]["item"].get("arguments") == "")

print()
print("[3] single-fragment call still emits exactly one delta")
one = [
    chunk({"tool_calls": [{"index": 0, "id": "call_12",
                           "function": {"name": "get_weather", "arguments": CITY_ARGS}}]}),
    chunk({}, "tool_calls"),
]
evs3 = events(one)
pieces3 = deltas_for(evs3, "response.function_call_arguments.delta")
check("one delta, not zero and not two", pieces3 == [CITY_ARGS], pieces3)

print()
print("[4] custom tool: the input fragments arrive whole too")
custom_stream = [
    chunk({"tool_calls": [{"index": 0, "id": "call_13",
                           "function": {"name": "apply_patch",
                                        "arguments": '{"input":"*** Begin'}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' Patch\\n+hi\\n*** End Patch"}'}}]}),
    chunk({}, "tool_calls"),
]
evs4 = events(custom_stream, {"usage": None, "custom_names": {"apply_patch"}})
pieces4 = deltas_for(evs4, "response.custom_tool_call_input.delta")
check("both fragments emitted", len(pieces4) == 2, pieces4)
try:
    rebuilt4 = json.loads("".join(pieces4))
except Exception as exc:
    rebuilt4 = {"error": repr(exc)}
check("input deltas concatenate to the full payload",
      rebuilt4.get("input") == "*** Begin Patch\n+hi\n*** End Patch", rebuilt4)
check("no function_call_arguments events for a custom tool",
      not [e for e in evs4 if e["type"].startswith("response.function_call_arguments")])

print()
print("[5] parallel calls: each one gets its own deltas, no cross-talk")
par = [
    chunk({"tool_calls": [
        {"index": 0, "id": "call_a", "function": {"name": "alpha", "arguments": '{"a"'}},
        {"index": 1, "id": "call_b", "function": {"name": "beta", "arguments": '{"b"'}},
    ]}),
    chunk({"tool_calls": [
        {"index": 0, "function": {"arguments": ':1}'}},
        {"index": 1, "function": {"arguments": ':2}'}},
    ]}),
    chunk({}, "tool_calls"),
]
evs5 = events(par)
check("call 0 rebuilds to {\"a\":1}", "".join(deltas_for(evs5, "response.function_call_arguments.delta", 0)) == '{"a":1}',
      deltas_for(evs5, "response.function_call_arguments.delta", 0))
check("call 1 rebuilds to {\"b\":2}", "".join(deltas_for(evs5, "response.function_call_arguments.delta", 1)) == '{"b":2}',
      deltas_for(evs5, "response.function_call_arguments.delta", 1))

print()
print("SUMMARY: PASS=%d FAIL=%d" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
