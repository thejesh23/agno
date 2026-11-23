import json

import pytest

from agno.models.message import Message
from agno.run.team import TeamRunOutput
from agno.session import TeamSession
from agno.team import _default_tools
from agno.team.team import Team


def _make_session() -> TeamSession:
    """Two team runs, each with a system prompt; the second answers through a tool call."""
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    runs = [
        [
            Message(role="system", content="SYS"),
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ],
        [
            Message(role="system", content="SYS"),
            Message(role="user", content="Q2"),
            Message(role="assistant", content=None, tool_calls=[tool_call]),
            Message(role="tool", content="ok", tool_call_id="call-1"),
            Message(role="assistant", content="A2"),
        ],
    ]
    return TeamSession(
        session_id="session-1",
        runs=[
            TeamRunOutput(run_id=f"run-{i}", session_id="session-1", team_id="team-1", messages=msgs)
            for i, msgs in enumerate(runs)
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "num_chats, expected",
    [
        (None, ["Q1", "A1", "Q2", "A2"]),
        (0, []),
        (1, ["Q2", "A2"]),
        (2, ["Q1", "A1", "Q2", "A2"]),
    ],
)
async def test_team_get_chat_history_counts_user_team_exchanges(async_mode, num_chats, expected):
    team = Team(id="team-1", members=[])
    tool = _default_tools._get_chat_history_function(team, _make_session(), async_mode=async_mode)

    raw = tool.entrypoint(num_chats=num_chats)
    if async_mode:
        raw = await raw

    assert [m.get("content") for m in json.loads(raw)] == expected
