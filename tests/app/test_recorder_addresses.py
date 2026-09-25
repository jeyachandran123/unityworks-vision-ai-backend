"""A recorder's IP address and its domain are two values, and one is dialled.

A site may be reachable at a local IP (`192.168.54.243`) on its own network
and at a domain (`site.freemyip.com`) from anywhere else. Both are kept.
`connect_via` names the one this server dials — chosen, never guessed between
two, and never silently swapped for the other.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from .conftest import bearer, make_user

BASE = "/api/v1/recorders"


@pytest.fixture
async def admin(seeded):
    async with seeded.state.database.session_scope() as session:
        _, user = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(user)
    return seeded


def _body(**overrides) -> dict:
    body = {
        "name": "Canteen DVR",
        "rtsp_port": 554,
        "username": "admin",
        "password": "Canteen-Pass-1",
        "brand": "hikvision",
    }
    body.update(overrides)
    return body


async def _post(client: AsyncClient, **overrides):
    return await client.post(
        BASE, json=_body(**overrides), headers=await bearer(client, "admin@example.com")
    )


async def test_only_a_domain_connects_by_domain(admin, client: AsyncClient):
    response = await _post(client, hostname="canteen.example.net")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["ip_address"], body["hostname"], body["connect_via"], body["address"]) == (
        "",
        "canteen.example.net",
        "hostname",
        "canteen.example.net",
    )


async def test_both_addresses_need_a_choice(admin, client: AsyncClient):
    """Two filled-in addresses and no choice is a question, not a guess."""
    unchosen = await _post(client, ip_address="192.168.54.243", hostname="canteen.example.net")
    assert unchosen.status_code == 422
    assert "choose which one" in unchosen.json()["message"]

    chosen = await _post(
        client,
        ip_address="192.168.54.243",
        hostname="canteen.example.net",
        connect_via="hostname",
    )
    assert chosen.status_code == 200, chosen.text
    assert chosen.json()["address"] == "canteen.example.net"
    assert chosen.json()["ip_address"] == "192.168.54.243", "the other address is kept"


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"hostname": "192.168.54.243"}, "enter it under IP address"),
        ({"ip_address": "canteen.example.net"}, "Domain / hostname"),
        ({"ip_address": "192.168.54.243", "connect_via": "hostname"}, "no"),
        ({"hostname": "canteen.example.net", "connect_via": "ip_address"}, "none is entered"),
        ({"ip_address": "192.168.54.243", "connect_via": "carrier-pigeon"}, "choose"),
    ],
)
async def test_each_address_mistake_says_how_to_fix_it(
    admin, client: AsyncClient, fields: dict, expected: str
):
    response = await _post(client, **fields)
    assert response.status_code == 422
    assert expected in response.json()["message"]


async def test_editing_keeps_the_choice_until_it_is_changed(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    created = (await _post(client, ip_address="192.168.54.243")).json()
    url = f"{BASE}/{created['id']}"

    # Adding the domain later does not change which address is dialled.
    added = await client.patch(url, json={"hostname": "canteen.example.net"}, headers=headers)
    assert added.status_code == 200, added.text
    assert (added.json()["connect_via"], added.json()["address"]) == (
        "ip_address",
        "192.168.54.243",
    )

    # Choosing the domain does.
    switched = (await client.patch(url, json={"connect_via": "hostname"}, headers=headers)).json()
    assert switched["address"] == "canteen.example.net"

    # Clearing the address in use leaves one choice, which is then the choice.
    cleared = await client.patch(url, json={"hostname": ""}, headers=headers)
    assert cleared.status_code == 200, cleared.text
    assert (cleared.json()["connect_via"], cleared.json()["address"]) == (
        "ip_address",
        "192.168.54.243",
    )

    # Clearing the last address is refused.
    emptied = await client.patch(url, json={"ip_address": ""}, headers=headers)
    assert emptied.status_code == 422
