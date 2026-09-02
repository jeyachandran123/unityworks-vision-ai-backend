"""The durable application domain.

`models` and `modules` are both imported here for their side effect: a table
that is not registered on `Base.metadata` is invisible to `create_all_for_tests`
and to Alembic autogenerate, and the failure mode of forgetting one is a
migration that proposes dropping a schema it simply could not see.
"""

from app.domain import models as _models  # noqa: F401
from app.domain import modules as _modules  # noqa: F401
