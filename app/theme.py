"""
Shared theme / UI helpers for the AI Agent Fellowship multipage app.
Both pages (AI Workspace, Document Intelligence) and the router import from here,
so the look-and-feel stays identical across the whole app.
"""

import os
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(ROOT, "assets", "logo.png")


def page_icon():
    """Favicon: the Nexus logo PNG, falling back to an emoji if missing."""
    return LOGO_PATH if os.path.exists(LOGO_PATH) else "🔷"


def logo_svg(size: int = 26, color: str | None = None) -> str:
    """Inline Nexus logo. Gradient by default; pass color='#fff' for a solid mark."""
    if color:
        stroke = fill = color
        defs = ""
    else:
        defs = ('<defs><linearGradient id="nxg" x1="0" y1="0" x2="1" y2="1">'
                '<stop offset="0" stop-color="#7c5cff"/>'
                '<stop offset="1" stop-color="#00d4ff"/></linearGradient></defs>')
        stroke = fill = "url(#nxg)"
    return f'''<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">{defs}
      <line x1="32" y1="33" x2="32" y2="14" stroke="{stroke}" stroke-width="3.4" stroke-linecap="round"/>
      <line x1="32" y1="33" x2="15" y2="49" stroke="{stroke}" stroke-width="3.4" stroke-linecap="round"/>
      <line x1="32" y1="33" x2="49" y2="49" stroke="{stroke}" stroke-width="3.4" stroke-linecap="round"/>
      <circle cx="32" cy="33" r="7" fill="{fill}"/><circle cx="32" cy="13" r="4.4" fill="{fill}"/>
      <circle cx="15" cy="49" r="4.4" fill="{fill}"/><circle cx="49" cy="49" r="4.4" fill="{fill}"/></svg>'''


def render_hero(title: str, subtitle: str, pill: str | None = None):
    """The gradient hero banner with a white Nexus mark."""
    pill_html = f'<span class="pill">{pill}</span>' if pill else ""
    st.markdown(f"""
    <div class="hero">
        <h1 style="display:flex;align-items:center;gap:12px;">{logo_svg(34, "#fff")}{title}</h1>
        <p>{subtitle}</p>{pill_html}
    </div>
    """, unsafe_allow_html=True)


def sidebar_brand(title: str, caption: str | None = None):
    """Sidebar brand block: gradient Nexus mark + title."""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:0 0 2px;">{logo_svg(26)}
      <span style="font-size:1.3rem;font-weight:800;color:var(--text);letter-spacing:-.3px;">{title}</span>
    </div>
    """, unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def appearance_toggle():
    """Dark/Light toggle. Uses one shared session_state key so it stays in sync
    across every page in the multipage app."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
    st.markdown('<div class="side-title">Appearance</div>', unsafe_allow_html=True)
    dm = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dm != st.session_state.dark_mode:
        st.session_state.dark_mode = dm
        st.rerun()


def inject_css(dark: bool):
    """Injects the full custom theme. Call once per page after set_page_config."""
    if dark:
        c = dict(bg="#0e1117", panel="#161a23", panel2="#1c2230", text="#e6e9ef",
                 muted="#9aa4b2", border="#262d3d", user="#1f6feb", user2="#388bfd",
                 bot="#20293a", accent="#7c5cff", accent2="#00d4ff")
    else:
        c = dict(bg="#eef2f8", panel="#ffffff", panel2="#ffffff", text="#1a1f2e",
                 muted="#5b6472", border="#c8d2e4", user="#2563eb", user2="#3b82f6",
                 bot="#ffffff", accent="#6d4bff", accent2="#0ea5e9")

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg:{c['bg']}; --panel:{c['panel']}; --panel2:{c['panel2']};
        --text:{c['text']}; --muted:{c['muted']}; --border:{c['border']};
        --user:{c['user']}; --user2:{c['user2']}; --bot:{c['bot']};
        --accent:{c['accent']}; --accent2:{c['accent2']};
    }}

    .stApp {{ background: var(--bg); color: var(--text);
        font-family:'Inter',sans-serif; }}
    /* Hide the menu/toolbar & footer, but KEEP the sidebar expand arrow usable */
    #MainMenu, footer {{ visibility:hidden; }}
    [data-testid="stToolbar"] {{ visibility:hidden; }}
    [data-testid="stDecoration"] {{ display:none; }}
    header[data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {{
        visibility:visible !important; display:flex !important; z-index:1000; }}
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stExpandSidebarButton"] button {{
        background:var(--panel2)!important; border:1px solid var(--border)!important;
        color:var(--accent)!important; border-radius:8px!important; }}
    .block-container {{ padding-top:1.2rem; max-width:1200px; }}

    /* ---------- Hero banner ---------- */
    .hero {{
        background: linear-gradient(120deg, var(--accent) 0%, var(--accent2) 100%);
        border-radius:18px; padding:22px 28px; margin-bottom:18px;
        box-shadow:0 10px 30px -10px rgba(124,92,255,.55); position:relative; overflow:hidden;
    }}
    .hero::after {{ content:""; position:absolute; top:-40%; right:-5%; width:220px; height:220px;
        background:rgba(255,255,255,.15); border-radius:50%; filter:blur(6px); }}
    .hero h1 {{ color:#fff; font-size:1.9rem; font-weight:800; margin:0; letter-spacing:-.5px; }}
    .hero p {{ color:rgba(255,255,255,.9); margin:.25rem 0 0; font-size:.95rem; font-weight:500; }}
    .hero .pill {{ display:inline-block; background:rgba(255,255,255,.2); color:#fff;
        padding:3px 12px; border-radius:20px; font-size:.72rem; font-weight:600; margin-top:10px;
        backdrop-filter:blur(6px); }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebarNav"] {{ display:none; }}
    section[data-testid="stSidebar"] {{ background:var(--panel); border-right:1px solid var(--border); }}
    section[data-testid="stSidebar"] * {{ color:var(--text); }}
    .side-card {{ background:var(--panel2); border:1px solid var(--border); border-radius:12px;
        padding:12px 14px; margin-bottom:12px; }}
    .side-title {{ font-size:.7rem; text-transform:uppercase; letter-spacing:1.2px;
        color:var(--muted); font-weight:700; margin-bottom:8px; }}

    /* ---------- Inputs ---------- */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"]>div {{
        background:var(--panel2)!important; color:var(--text)!important;
        border:1px solid var(--border)!important; border-radius:10px!important; }}
    .stTextInput input:focus, .stTextArea textarea:focus {{ border-color:var(--accent)!important;
        box-shadow:0 0 0 2px rgba(124,92,255,.25)!important; }}
    /* Robust selectbox theming (version-agnostic — works on Streamlit Cloud too) */
    [data-baseweb="select"] > div {{
        background:var(--panel2)!important; border:1px solid var(--border)!important;
        border-radius:10px!important; }}
    [data-baseweb="select"] div, [data-baseweb="select"] span,
    [data-baseweb="select"] input {{ color:var(--text)!important; }}
    [data-baseweb="select"] svg {{ fill:var(--muted)!important; }}
    /* Dropdown menu options popup */
    [data-baseweb="popover"] div, [data-baseweb="menu"], [role="listbox"] {{
        background:var(--panel)!important; }}
    [role="option"] {{ background:var(--panel)!important; color:var(--text)!important; }}
    [role="option"]:hover {{ background:var(--panel2)!important; }}
    /* Placeholder text visible in both themes */
    input::placeholder, textarea::placeholder {{ color:var(--muted)!important; opacity:1!important; }}

    /* ---------- Buttons (incl. form-submit & file-uploader browse) ---------- */
    .stButton>button, .stDownloadButton>button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploaderDropzone"] button {{
        background:var(--panel2)!important; color:var(--text)!important;
        border:1px solid var(--border)!important;
        border-radius:10px!important; font-weight:600; transition:all .15s ease; }}
    .stButton>button:hover, .stDownloadButton>button:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover {{
        border-color:var(--accent)!important; color:var(--accent)!important;
        transform:translateY(-1px); }}
    /* Form container should follow the theme too */
    [data-testid="stForm"] {{ background:transparent!important;
        border:1px solid var(--border)!important; border-radius:12px!important; }}

    /* ---------- Metric cards ---------- */
    .metric-row {{ display:flex; gap:10px; margin-bottom:6px; }}
    .metric {{ flex:1; background:var(--panel2); border:1px solid var(--border);
        border-radius:12px; padding:10px 12px; text-align:center;
        box-shadow:0 1px 3px rgba(2,6,23,.06); }}
    .metric .v {{ font-size:1.15rem; font-weight:800;
        background:linear-gradient(90deg,var(--accent),var(--accent2));
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    .metric .l {{ font-size:.66rem; color:var(--muted); text-transform:uppercase;
        letter-spacing:.5px; font-weight:600; margin-top:2px; }}

    /* ---------- Nav cards (router landing) ---------- */
    .nav-card {{ background:var(--panel2); border:1px solid var(--border); border-radius:16px;
        padding:20px 22px; height:100%; transition:all .15s ease; }}
    .nav-card:hover {{ border-color:var(--accent); transform:translateY(-2px); }}
    .nav-card h3 {{ margin:6px 0 4px; font-size:1.15rem; font-weight:800; color:var(--text); }}
    .nav-card p {{ margin:0; color:var(--muted); font-size:.9rem; }}
    .nav-badge {{ display:inline-block; font-size:.68rem; font-weight:700; padding:2px 10px;
        border-radius:20px; background:rgba(124,92,255,.15); color:var(--accent); }}

    /* ---------- Document library rows / status ---------- */
    .doc-row {{ background:var(--panel2); border:1px solid var(--border); border-radius:10px;
        padding:8px 10px; margin-bottom:6px; }}
    .doc-name {{ font-size:.85rem; font-weight:700; color:var(--text); word-break:break-word; }}
    .doc-meta {{ font-size:.72rem; color:var(--muted); margin-top:3px; }}
    .pill-ok {{ display:inline-block; font-size:.58rem; font-weight:700; padding:1px 8px;
        border-radius:20px; background:rgba(34,197,94,.18); color:#16a34a; }}
    .match-badge {{ display:inline-block; font-size:.56rem; font-weight:700; padding:1px 7px;
        border-radius:20px; background:rgba(14,165,233,.18); color:var(--accent2); margin-left:6px; }}
    .tool-card {{ background:var(--panel2); border:1px solid var(--border); border-radius:12px;
        padding:14px 16px; margin-bottom:10px; color:var(--text); }}
    .tool-card h4 {{ margin:0 0 6px; font-size:.95rem; font-weight:800; color:var(--text); }}
    .stat-mini {{ font-size:.72rem; color:var(--muted); }}
    /* Login card */
    .login-wrap {{ max-width:420px; margin:6vh auto 0; }}

    /* ---------- Citation / source cards ---------- */
    .cite-card {{ background:var(--panel2); border:1px solid var(--border);
        border-left:3px solid var(--accent); border-radius:10px; padding:10px 14px; margin:8px 0 2px; }}
    .cite-card .ct {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.9px;
        color:var(--muted); font-weight:700; margin-bottom:6px; }}
    .cite-item {{ font-size:.86rem; color:var(--text); padding:6px 0; border-top:1px solid var(--border); }}
    .cite-item:first-of-type {{ border-top:none; }}
    .cite-src {{ font-weight:700; }}
    .cite-badge {{ font-size:.6rem; font-weight:700; color:var(--accent);
        background:rgba(124,92,255,.16); padding:1px 8px; border-radius:20px; margin-left:8px; }}
    .cite-snip {{ color:var(--muted); font-size:.78rem; margin-top:3px; line-height:1.4; }}

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {{ background:var(--bot); border:1px solid var(--border);
        border-radius:14px; padding:6px 14px; margin:8px 0; box-shadow:0 2px 8px -4px rgba(0,0,0,.2); }}
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ color:var(--text); }}
    [data-testid="stChatMessage"] code {{ font-family:'JetBrains Mono',monospace;
        background:rgba(124,92,255,.12); padding:2px 6px; border-radius:6px; }}
    [data-testid="stChatInput"] textarea {{ background:var(--panel2)!important; color:var(--text)!important; }}
    [data-testid="stBottom"], [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"], [data-testid="stChatInput"] {{ background:var(--bg)!important; }}
    [data-testid="stChatInput"] {{ border:none!important; box-shadow:none!important; }}
    [data-testid="stChatInput"] > div {{ background:var(--panel2)!important;
        border:1px solid var(--border)!important; border-radius:12px!important; }}

    /* ---------- Empty state ---------- */
    .empty {{ text-align:center; padding:52px 20px; color:var(--muted); }}
    .empty .big {{ font-size:3rem; margin-bottom:8px; }}
    .chip {{ display:inline-block; background:var(--panel2); border:1px solid var(--border);
        color:var(--text); padding:8px 14px; border-radius:22px; margin:5px; font-size:.85rem; font-weight:500; }}
    .stMetric {{ background:var(--panel2); border:1px solid var(--border); border-radius:12px; padding:8px 12px; }}

    /* ---------- Widget theming (light & dark) ---------- */
    hr {{ border:none!important; border-top:1px solid var(--border)!important; opacity:1!important; margin:.6rem 0!important; }}
    [data-baseweb="checkbox"] > div:first-child {{ border:1px solid var(--border)!important; }}
    [data-baseweb="checkbox"] > div:first-child > div {{ box-shadow:0 1px 2px rgba(2,6,23,.35)!important; }}
    .stTextInput [data-baseweb="base-input"], .stTextInput [data-baseweb="input"] {{
        background:var(--panel2)!important; }}
    .stTextInput button {{ background:transparent!important; color:var(--muted)!important; border:none!important; }}
    [data-testid="stChatInputSubmitButton"] {{ color:var(--accent)!important; background:transparent!important; }}
    [data-testid="stChatInputSubmitButton"]:hover {{ background:rgba(124,92,255,.12)!important; }}
    .stButton>button:disabled, .stDownloadButton>button:disabled {{
        background:var(--panel2)!important; color:var(--muted)!important;
        border:1px solid var(--border)!important; opacity:.75!important; }}
    /* File uploader (Document Intelligence) */
    [data-testid="stFileUploaderDropzone"] {{ background:var(--panel2)!important;
        border:1px dashed var(--border)!important; }}
    </style>
    """, unsafe_allow_html=True)
