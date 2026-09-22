"""Two admin tiers, and no third.

The product has a Platform Admin, who is not a role at all (a row in
`platform_operator_grants`), and an Organization Admin, who is the top tenant
role. The tenant `super_admin` role that sat between them is gone: it read as a
third tier, and its label was the one the platform console showed for the
operator, which is exactly the ambiguity these tests keep closed.
"""

from __future__ import annotations

from app.authorization.model import ROLE_PERMISSIONS, Permission, Role


def test_there_is_no_super_admin_role():
    assert "super_admin" not in {role.value for role in Role}


def test_organization_admin_holds_every_permission():
    """Full authority inside every organization the admin holds.

    Stated as the complete set rather than enumerated, so a permission added
    later reaches the Organization Admin without anybody remembering to list it.
    """
    assert ROLE_PERMISSIONS[Role.ORG_ADMIN] == frozenset(Permission)


def test_organization_admin_reaches_the_engineering_surfaces():
    assert Role.ORG_ADMIN.is_platform_role
    assert Role.DEVELOPER.is_platform_role
    assert not Role.RESTAURANT_MANAGER.is_platform_role


def test_every_role_has_a_permission_set():
    assert set(ROLE_PERMISSIONS) == set(Role)
