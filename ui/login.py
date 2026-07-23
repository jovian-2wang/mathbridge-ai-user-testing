import html
import json
import os
from pathlib import Path

import streamlit as st


DEMO_USERS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "memory"
    / "demo_users.json"
)


DEFAULT_DEMO_USERS = {
    "alex": {
        "password": "1234",
        "role": "Student",
        "display_name": "Alex",
        "memory_file": "alex.json",
    },
    "liam": {
        "password": "1234",
        "role": "Student",
        "display_name": "Liam",
        "memory_file": "liam.json",
    },
    "maya": {
        "password": "1234",
        "role": "Student",
        "display_name": "Maya",
        "memory_file": "maya.json",
    },
    "teacher_demo": {
        "password": "1234",
        "role": "Teacher",
        "display_name": "Demo Teacher",
    },
    "parent_demo": {
        "password": "1234",
        "role": "Parent",
        "display_name": "Alex's Parent",
        "children": [
            {
                "name": "Alex",
                "memory_file": "alex.json",
            }
        ],
    },
    "alex_parent": {
        "password": "1234",
        "role": "Parent",
        "display_name": "Alex's Parent",
        "children": [
            {
                "name": "Alex",
                "memory_file": "alex.json",
            }
        ],
    },
    "liam_parent": {
        "password": "1234",
        "role": "Parent",
        "display_name": "Liam's Parent",
        "children": [
            {
                "name": "Liam",
                "memory_file": "liam.json",
            }
        ],
    },
    "maya_parent": {
        "password": "1234",
        "role": "Parent",
        "display_name": "Maya's Parent",
        "children": [
            {
                "name": "Maya",
                "memory_file": "maya.json",
            }
        ],
    },
}


_LOGIN_STYLE = """
<style>
.mathbridge-login-hero {
    border: 1px solid #dbeafe;
    border-radius: 24px;
    padding: 30px 34px;
    margin: 6px 0 22px 0;
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 34%),
        linear-gradient(135deg, #f8fbff 0%, #ffffff 54%, #eef6ff 100%);
}
.mathbridge-login-eyebrow {
    color: #2563eb;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.mathbridge-login-title {
    color: #111827;
    font-size: 2.15rem;
    line-height: 1.08;
    font-weight: 800;
    margin: 0 0 12px 0;
}
.mathbridge-login-subtitle {
    color: #475569;
    font-size: 1.03rem;
    line-height: 1.65;
    max-width: 920px;
    margin: 0;
}
.mathbridge-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 20px;
}
.mathbridge-pill {
    border: 1px solid #bfdbfe;
    background: #eff6ff;
    color: #1d4ed8;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 0.86rem;
    font-weight: 700;
}
.mathbridge-account-line {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 12px 14px;
    background: #f9fafb;
    margin-bottom: 8px;
}
.mathbridge-account-line strong {
    color: #111827;
}
.mathbridge-muted {
    color: #6b7280;
}
.mathbridge-testing-panel {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px 20px;
    background: #ffffff;
    margin: 10px 0 18px 0;
}
.mathbridge-testing-panel h3 {
    margin: 0 0 6px 0;
    color: #111827;
}
.mathbridge-testing-panel p {
    color: #4b5563;
    margin: 0 0 12px 0;
}
.mathbridge-account-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}
.mathbridge-test-account-card {
    border: 1px solid #dbeafe;
    border-radius: 16px;
    padding: 14px 14px;
    background: #f8fbff;
}
.mathbridge-test-account-card h4 {
    margin: 0 0 8px 0;
    color: #111827;
}
.mathbridge-test-account-card code {
    background: #eff6ff;
    color: #1d4ed8;
    border-radius: 6px;
    padding: 2px 5px;
}
.mathbridge-contact-box {
    border: 1px solid #fecaca;
    border-radius: 14px;
    padding: 12px 14px;
    background: #fff7f7;
    color: #991b1b;
    margin-top: 12px;
    font-weight: 700;
}
@media (max-width: 900px) {
    .mathbridge-account-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


STUDENT_RUNTIME_KEYS = [
    "active_student_id",
    "student_id",
    "student_selector_v2",
    "student_runtime_owner",
    "chat_history",
    "signals",
    "curriculum_evidence",
    "session_saved",
    "hint_level",
    "current_problem_text",
    "problem_solved",
    "current_topic",
    "student_session_cache",
    "context_used",
    "last_context_used",
    "last_timings",
]


APP_AUTH_KEYS = [
    "logged_in",
    "username",
    "authentication_status",
    "name",
    "email",
    "role",
    "display_name",
    "memory_file",
    "children",
    "view",
]


def _escape(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _get_setting(name: str, default: str) -> str:
    """Read settings from Streamlit secrets first, then environment."""
    secret_value = None

    try:
        secret_value = st.secrets.get(name, None)
    except Exception:
        secret_value = None

    return str(os.getenv(name, secret_value or default))


def _public_testing_mode() -> bool:
    return _get_setting("MATHBRIDGE_PUBLIC_TESTING_MODE", "1") == "1"


def _show_demo_shortcuts() -> bool:
    return _get_setting("MATHBRIDGE_SHOW_DEMO_SHORTCUTS", "0") == "1"


def _support_email() -> str:
    return _get_setting("MATHBRIDGE_SUPPORT_EMAIL", "jiangweiwang@ufl.edu")


def save_demo_users(users):
    DEMO_USERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DEMO_USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def load_demo_users():
    DEMO_USERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DEMO_USERS_PATH.exists():
        save_demo_users(DEFAULT_DEMO_USERS)
        return dict(DEFAULT_DEMO_USERS)

    with open(DEMO_USERS_PATH, "r", encoding="utf-8") as f:
        users = json.load(f)

    # Keep existing demo files backward-compatible without overwriting
    # any password the user may have changed during testing.
    changed = False
    for username, default_record in DEFAULT_DEMO_USERS.items():
        if username not in users:
            users[username] = default_record
            changed = True

    if changed:
        save_demo_users(users)

    return users


def _credentials_from_users(users: dict) -> dict:
    credentials = {"usernames": {}}

    for username, record in users.items():
        if not isinstance(record, dict):
            continue

        display_name = str(record.get("display_name") or username)
        name_parts = display_name.split(maxsplit=1)
        first_name = name_parts[0] if name_parts else username
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        credentials["usernames"][username] = {
            "email": record.get("email", f"{username}@mathbridge.local"),
            "name": display_name,
            "first_name": first_name,
            "last_name": last_name,
            "password": record.get("password", "1234"),
            "roles": [record.get("role", "Student")],
        }

    return credentials


def _make_authenticator(users: dict):
    try:
        import streamlit_authenticator as stauth
    except Exception as exc:
        st.error(
            "Missing dependency: streamlit-authenticator. "
            "Run `pip install -r requirements.txt` and restart Streamlit."
        )
        st.exception(exc)
        st.stop()

    cookie_name = _get_setting(
        "MATHBRIDGE_AUTH_COOKIE_NAME",
        "mathbridge_user_testing_auth",
    )
    cookie_key = _get_setting(
        "MATHBRIDGE_AUTH_COOKIE_KEY",
        "replace_this_with_a_long_random_secret_before_deployment",
    )
    cookie_days = float(
        _get_setting("MATHBRIDGE_AUTH_COOKIE_DAYS", "30")
    )

    return stauth.Authenticate(
        _credentials_from_users(users),
        cookie_name=cookie_name,
        cookie_key=cookie_key,
        cookie_expiry_days=cookie_days,
        auto_hash=True,
    )


def init_auth_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "username" not in st.session_state:
        st.session_state.username = None

    if "role" not in st.session_state:
        st.session_state.role = None

    if "display_name" not in st.session_state:
        st.session_state.display_name = None

    if "memory_file" not in st.session_state:
        st.session_state.memory_file = None

    if "children" not in st.session_state:
        st.session_state.children = []


def _clear_student_runtime_state():
    """Clear temporary Student Chat state when switching demo users."""
    for key in STUDENT_RUNTIME_KEYS:
        if key in st.session_state:
            del st.session_state[key]


def _clear_app_login_state():
    """Clear local Streamlit login/session state after cookie logout."""
    _clear_student_runtime_state()

    for key in APP_AUTH_KEYS:
        if key in st.session_state:
            del st.session_state[key]

    # Recreate the keys that the rest of the app expects to exist.
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.display_name = None
    st.session_state.memory_file = None
    st.session_state.children = []
    st.session_state.view = "landing"


def _set_logged_in_user(username: str, user_record: dict):
    old_username = st.session_state.get("username")
    old_memory_file = st.session_state.get("memory_file")
    new_memory_file = user_record.get("memory_file")

    if old_username != username or old_memory_file != new_memory_file:
        _clear_student_runtime_state()

    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.role = user_record["role"]
    st.session_state.display_name = user_record["display_name"]
    st.session_state.memory_file = new_memory_file
    st.session_state.children = user_record.get("children", [])

    if "view" not in st.session_state or st.session_state.view is None:
        st.session_state.view = "landing"

    if user_record["role"] == "Student":
        if new_memory_file:
            student_id = new_memory_file.replace(".json", "")
        else:
            student_id = username

        st.session_state.active_student_id = student_id
        st.session_state.student_id = student_id
        st.session_state.student_selector_v2 = student_id


def _render_hero():
    st.markdown(_LOGIN_STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <section class="mathbridge-login-hero">
          <div class="mathbridge-login-eyebrow">User Testing Release · Persistent Login</div>
          <h1 class="mathbridge-login-title">MathBridge AI</h1>
          <p class="mathbridge-login-subtitle">
            A contextualized, curriculum-grounded math tutoring platform for student,
            teacher, parent, and class-level learning support. Use one of the test
            accounts below to sign in. Your browser can keep you signed in for repeated
            testing, so you do not need to install anything locally.
          </p>
          <div class="mathbridge-pill-row">
            <span class="mathbridge-pill">Student Profile</span>
            <span class="mathbridge-pill">Contextualized Tutoring</span>
            <span class="mathbridge-pill">Curriculum Grounding</span>
            <span class="mathbridge-pill">Learning Signals</span>
            <span class="mathbridge-pill">Teacher + Parent Loop</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _first_available_user(
    users: dict,
    role: str,
    preferred: tuple[str, ...],
) -> str | None:
    for username in preferred:
        if users.get(username, {}).get("role") == role:
            return username

    for username, record in users.items():
        if isinstance(record, dict) and record.get("role") == role:
            return username

    return None


def _user_label(users: dict, username: str | None) -> str:
    if not username or username not in users:
        return "Not available"

    record = users[username]
    display_name = record.get("display_name", username)
    return f"{display_name} · @{username}"


def _render_demo_shortcuts(users: dict):
    if not _show_demo_shortcuts():
        return

    st.info(
        "Presentation shortcuts are visible because "
        "MATHBRIDGE_SHOW_DEMO_SHORTCUTS=1. Hide them for student/parent testing."
    )

    student_user = _first_available_user(
        users,
        "Student",
        ("maya", "liam", "alex"),
    )
    teacher_user = _first_available_user(
        users,
        "Teacher",
        ("teacher_demo",),
    )
    parent_user = _first_available_user(
        users,
        "Parent",
        ("parent_demo", "alex_parent", "liam_parent", "maya_parent"),
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            f"Student Demo ({_user_label(users, student_user)})",
            use_container_width=True,
            disabled=student_user is None,
            key="start_student_demo",
        ):
            _set_logged_in_user(student_user, users[student_user])
            st.rerun()

    with col2:
        if st.button(
            f"Teacher Demo ({_user_label(users, teacher_user)})",
            use_container_width=True,
            disabled=teacher_user is None,
            key="start_teacher_demo",
        ):
            _set_logged_in_user(teacher_user, users[teacher_user])
            st.rerun()

    with col3:
        if st.button(
            f"Parent Demo ({_user_label(users, parent_user)})",
            use_container_width=True,
            disabled=parent_user is None,
            key="start_parent_demo",
        ):
            _set_logged_in_user(parent_user, users[parent_user])
            st.rerun()


def _role_order(role: str) -> int:
    order = {
        "Student": 0,
        "Teacher": 1,
        "Parent": 2,
    }
    return order.get(role, 9)


def _render_account_reference(users: dict):
    rows = []
    for username, record in users.items():
        if not isinstance(record, dict):
            continue

        role = record.get("role", "Demo")
        display_name = record.get("display_name", username)
        detail = ""

        if record.get("memory_file"):
            detail = f"Memory: {record['memory_file']}"
        elif record.get("children"):
            child_names = [
                str(child.get("name") or child.get("student_id") or child)
                for child in record.get("children", [])
            ]
            detail = "Children: " + ", ".join(child_names)

        rows.append(
            {
                "username": username,
                "role": role,
                "display_name": display_name,
                "detail": detail,
            }
        )

    rows.sort(key=lambda item: (_role_order(item["role"]), item["username"]))

    st.caption("Available test accounts")
    for row in rows:
        detail = f"<br><span class='mathbridge-muted'>{_escape(row['detail'])}</span>" if row["detail"] else ""
        st.markdown(
            f"""
            <div class="mathbridge-account-line">
              <strong>{_escape(row['display_name'])}</strong>
              <span class="mathbridge-muted"> · {row['role']} · @{_escape(row['username'])}</span>
              {detail}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _record_password(record: dict) -> str:
    return str(record.get("password", "1234"))


def _first_role_record(users: dict, role: str, preferred: tuple[str, ...]) -> tuple[str | None, dict]:
    username = _first_available_user(users, role, preferred)
    if username and username in users:
        return username, users[username]
    return None, {}


def _render_test_account_card(role_label: str, description: str, username: str | None, record: dict):
    if not username:
        account_html = "<span class='mathbridge-muted'>Not configured</span>"
    else:
        account_html = (
            f"<div><strong>Account:</strong> <code>{_escape(username)}</code></div>"
            f"<div><strong>Password:</strong> <code>{_escape(_record_password(record))}</code></div>"
        )

    st.markdown(
        f"""
        <div class="mathbridge-test-account-card">
          <h4>{_escape(role_label)}</h4>
          <p>{_escape(description)}</p>
          {account_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_public_testing_info(users: dict):
    """Show user-facing test accounts and support contact on the login page."""
    student_user, student_record = _first_role_record(
        users,
        "Student",
        ("alex", "liam", "maya"),
    )
    teacher_user, teacher_record = _first_role_record(
        users,
        "Teacher",
        ("teacher_demo",),
    )
    parent_user, parent_record = _first_role_record(
        users,
        "Parent",
        ("alex_parent", "parent_demo", "liam_parent", "maya_parent"),
    )

    st.markdown(
        f"""
        <section class="mathbridge-testing-panel">
          <h3>Testing accounts</h3>
          <p>
            This is an online user-testing version. Open this link in a browser,
            sign in with one of the accounts below, and continue using the same
            browser later. No terminal commands or local installation are needed.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_test_account_card(
            "Student login",
            "Use this account to try the tutoring chat and learner profile.",
            student_user,
            student_record,
        )
    with c2:
        _render_test_account_card(
            "Teacher login",
            "Use this account to review teacher insights and class overview.",
            teacher_user,
            teacher_record,
        )
    with c3:
        _render_test_account_card(
            "Parent login",
            "Use this account to view weekly summary and home practice.",
            parent_user,
            parent_record,
        )

    contact = _support_email()
    st.markdown(
        f"""
        <div class="mathbridge-contact-box">
          If you run into any issue, please contact: {_escape(contact)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login():
    init_auth_state()

    users = load_demo_users()
    authenticator = _make_authenticator(users)

    _render_hero()
    _render_public_testing_info(users)

    st.subheader("Sign in")
    st.caption(
        "Use one of the testing accounts above. "
        "Login is remembered on this browser until the authentication cookie expires."
    )

    try:
        authenticator.login(
            location="main",
            fields={
                "Form name": "MathBridge AI Login",
                "Username": "Username",
                "Password": "Password",
                "Login": "Log in",
            },
            key="mathbridge_auth_login",
            clear_on_submit=False,
        )
    except Exception as exc:
        st.error("Login failed to render.")
        st.exception(exc)
        st.stop()

    auth_status = st.session_state.get("authentication_status")
    auth_username = st.session_state.get("username")

    if auth_status:
        if auth_username in users:
            _set_logged_in_user(auth_username, users[auth_username])
            st.rerun()

        st.error("This authenticated username is not registered in MathBridge.")
        st.stop()

    if auth_status is False:
        st.error("Invalid username or password.")
    else:
        st.info("Please sign in to continue.")

    _render_demo_shortcuts(users)

    if not _public_testing_mode():
        with st.expander("Local presentation tools", expanded=False):
            _render_account_reference(users)
            st.caption(
                "Default demo passwords are 1234 unless you changed them in data/memory/demo_users.json."
            )


def require_login():
    init_auth_state()

    if not st.session_state.logged_in:
        render_login()
        st.stop()


def render_user_panel():
    init_auth_state()

    users = load_demo_users()

    with st.sidebar:
        st.markdown("### MathBridge AI")
        st.caption("Contextualized learning demo")
        st.divider()

        st.markdown("### Current User")
        st.write(f"**Name:** {st.session_state.display_name}")
        st.write(f"**Role:** {st.session_state.role}")

        if st.session_state.memory_file:
            st.write(f"**Memory:** `{st.session_state.memory_file}`")

        if st.session_state.children:
            child_names = []
            for child in st.session_state.children:
                if isinstance(child, dict):
                    child_names.append(
                        str(child.get("name") or child.get("student_id") or "Child")
                    )
                else:
                    child_names.append(str(child))
            st.write(f"**Linked child:** {', '.join(child_names)}")

        # Important: keep exactly one logout control.
        # streamlit-authenticator must own this button so it can clear the
        # persistent browser cookie. The callback clears MathBridge's local
        # session state after the cookie logout runs.
        try:
            authenticator = _make_authenticator(users)
            authenticator.logout(
                button_name="Log out",
                location="sidebar",
                key="sidebar_logout",
                callback=_clear_app_login_state,
            )
        except Exception as exc:
            st.warning(
                "Logout could not be rendered. Refresh the page or clear this "
                "site's cookies if you need to switch accounts."
            )
            st.caption(str(exc))
