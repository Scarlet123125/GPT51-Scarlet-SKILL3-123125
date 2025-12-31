import streamlit as st
import yaml
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import io
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

class ModelConfig:
    """Supported LLM models configuration"""
    MODELS = [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "claude-3.5-sonnet",
        "claude-3.5-haiku",
        "grok-4-fast-reasoning",
        "grok-3-mini",
    ]

    @staticmethod
    def get_provider(model: str) -> str:
        """Determine provider from model name"""
        m = model.lower()
        if "gpt" in m:
            return "openai"
        elif "gemini" in m:
            return "gemini"
        elif "claude" in m:
            return "anthropic"
        elif "grok" in m:
            return "grok"
        return "unknown"


class UIConfig:
    """UI configuration including themes and styles"""
    # 20 wow painter styles
    PAINTER_STYLES = [
        "Van Gogh", "Monet", "Picasso", "Da Vinci", "Rembrandt",
        "Vermeer", "Caravaggio", "Matisse", "Kandinsky", "Pollock",
        "Rothko", "Warhol", "Klimt", "Munch", "Degas",
        "Renoir", "Cézanne", "Gauguin", "Hokusai", "Turner"
    ]

    STYLE_CSS = {
        "Van Gogh": "background: radial-gradient(circle at top left, #243B55, #141E30);",
        "Monet": "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);",
        "Picasso": "background: linear-gradient(to right, #fa709a 0%, #fee140 100%);",
        "Da Vinci": "background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);",
        "Rembrandt": "background: linear-gradient(to top, #30cfd0 0%, #330867 100%);",
        "Vermeer": "background: linear-gradient(135deg, #fdfcfb 0%, #e2d1f9 100%);",
        "Caravaggio": "background: radial-gradient(circle at top, #2c3e50 0%, #000000 60%);",
        "Matisse": "background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);",
        "Kandinsky": "background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);",
        "Pollock": "background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%);",
        "Rothko": "background: linear-gradient(135deg, #f85032 0%, #e73827 100%);",
        "Warhol": "background: linear-gradient(135deg, #fceabb 0%, #f8b500 100%);",
        "Klimt": "background: linear-gradient(135deg, #f6e27a 0%, #f0a830 100%);",
        "Munch": "background: linear-gradient(135deg, #2980b9 0%, #6dd5fa 100%);",
        "Degas": "background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);",
        "Renoir": "background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);",
        "Cézanne": "background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);",
        "Gauguin": "background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);",
        "Hokusai": "background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);",
        "Turner": "background: linear-gradient(135deg, #f3904f 0%, #3b4371 100%);",
    }


# ============================================================================
# LOCALIZATION
# ============================================================================

LABELS = {
    "Dashboard": {"English": "📊 Dashboard", "繁體中文": "📊 儀表板"},
    "510k_tab": {"English": "🔍 510(k) Intelligence", "繁體中文": "🔍 510(k) 智能分析"},
    "pdf_tab": {"English": "📄 PDF → Markdown", "繁體中文": "📄 PDF → Markdown"},
    "summary_tab": {"English": "📝 Summary & Entities", "繁體中文": "📝 摘要與實體"},
    "diff_tab": {"English": "🔄 Comparator", "繁體中文": "🔄 文件比較"},
    "checklist_tab": {"English": "✅ Checklist & Report", "繁體中文": "✅ 檢查清單與報告"},
    "notes_tab": {"English": "📓 Note Keeper & Magics", "繁體中文": "📓 筆記管理與魔法工具"},
    "orch_tab": {"English": "🎼 Orchestration", "繁體中文": "🎼 協調編排"},
    "dynamic_tab": {"English": "🤖 Dynamic Agents", "繁體中文": "🤖 動態代理生成"},
    "config_files_tab": {"English": "🧩 Config & Files", "繁體中文": "🧩 設定與檔案"},
    "Run Agent": {"English": "▶️ Run Agent", "繁體中文": "▶️ 執行代理"},
    "Model": {"English": "Model", "繁體中文": "模型"},
    "Max Tokens": {"English": "Max Tokens", "繁體中文": "最大標記數"},
    "Temperature": {"English": "Temperature", "繁體中文": "溫度"},
}


def t(key: str) -> str:
    """Translate label based on current language"""
    lang = st.session_state.get("language", "English")
    return LABELS.get(key, {}).get(lang, key)


# ============================================================================
# LLM ROUTER
# ============================================================================

def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 12000,
    temperature: float = 0.2,
    api_keys: Optional[Dict[str, str]] = None
) -> str:
    """
    Unified LLM interface supporting OpenAI, Gemini, Anthropic, Grok
    """
    provider = ModelConfig.get_provider(model)

    # Get API keys
    if api_keys is None:
        api_keys = st.session_state.get("api_keys", {})

    try:
        if provider == "openai":
            return call_openai(model, system_prompt, user_prompt, max_tokens, temperature, api_keys)
        elif provider == "gemini":
            return call_gemini(model, system_prompt, user_prompt, max_tokens, temperature, api_keys)
        elif provider == "anthropic":
            return call_anthropic(model, system_prompt, user_prompt, max_tokens, temperature, api_keys)
        elif provider == "grok":
            return call_grok(model, system_prompt, user_prompt, max_tokens, temperature, api_keys)
        else:
            raise ValueError(f"Unknown provider for model: {model}")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {str(e)}")


def call_openai(model, system_prompt, user_prompt, max_tokens, temperature, api_keys):
    """Call OpenAI API"""
    api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except ImportError:
        raise RuntimeError("OpenAI package not installed. Run: pip install openai")


def call_gemini(model, system_prompt, user_prompt, max_tokens, temperature, api_keys):
    """Call Google Gemini API"""
    api_key = api_keys.get("gemini") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found")

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model_instance = genai.GenerativeModel(model)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = model_instance.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
        return response.text
    except ImportError:
        raise RuntimeError("Google Generative AI package not installed. Run: pip install google-generativeai")


def call_anthropic(model, system_prompt, user_prompt, max_tokens, temperature, api_keys):
    """Call Anthropic Claude API"""
    api_key = api_keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not found")

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.content[0].text
    except ImportError:
        raise RuntimeError("Anthropic package not installed. Run: pip install anthropic")


def call_grok(model, system_prompt, user_prompt, max_tokens, temperature, api_keys):
    """Call xAI Grok API"""
    api_key = api_keys.get("grok") or os.getenv("GROK_API_KEY")
    if not api_key:
        raise ValueError("Grok API key not found")

    try:
        import httpx

        response = httpx.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except ImportError:
        raise RuntimeError("HTTPX package not installed. Run: pip install httpx")


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def extract_pdf_pages_to_text(file, start_page: int, end_page: int) -> str:
    """Extract text from PDF using pypdf (1-based indexing)"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file)
        total_pages = len(reader.pages)

        # Validate page range
        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, end_page)

        text_parts = []
        for i in range(start_idx, end_idx):
            page = reader.pages[i]
            text_parts.append(page.extract_text())

        return "\n\n".join(text_parts)
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""


def extract_docx_to_text(file) -> str:
    """Extract text from DOCX using python-docx"""
    try:
        from docx import Document

        doc = Document(file)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")
    except Exception as e:
        st.error(f"DOCX extraction error: {str(e)}")
        return ""


# ============================================================================
# AGENT EXECUTION ENGINE
# ============================================================================

def load_agents_config() -> Dict:
    """Load agents configuration from session state or default file"""
    if "agents_cfg" in st.session_state:
        return st.session_state["agents_cfg"]

    # Try to load from agents.yaml file
    if os.path.exists("agents.yaml"):
        with open("agents.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            if "agents" not in config:
                config["agents"] = {}
            st.session_state["agents_cfg"] = config
            return config

    # Return minimal default if no file found
    config = {"agents": {}}
    st.session_state["agents_cfg"] = config
    return config


def agent_run_ui(
    agent_id: str,
    tab_key: str,
    default_prompt: str = "",
    default_input_text: str = "",
    allow_model_override: bool = True,
    tab_label_for_history: Optional[str] = None,
    override_prompt: Optional[str] = None,
):
    """
    Reusable agent execution interface

    Features:
    - Status indicators
    - Model / temperature / max_tokens (default 12000) controls
    - Editable prompt
    - Editable input
    - Editable output with markdown preview
    - Ability to send output to a global clipboard for chaining to the next agent
    """
    agents_cfg = load_agents_config()
    agent_cfg = agents_cfg.get("agents", {}).get(agent_id, {})

    if not agent_cfg:
        st.error(f"Agent '{agent_id}' not found in configuration")
        return

    # Agent info
    st.markdown(f"### {agent_cfg.get('name', agent_id)}")
    st.caption(agent_cfg.get('description', ''))

    # Status indicator
    status_key = f"{tab_key}_status"
    if status_key not in st.session_state:
        st.session_state[status_key] = "pending"

    status = st.session_state[status_key]
    status_colors = {
        "pending": "🔵",
        "running": "🟡",
        "done": "🟢",
        "error": "🔴"
    }
    st.info(f"Status: {status_colors.get(status, '⚪')} {status}")

    # Configuration
    col1, col2, col3 = st.columns(3)

    with col1:
        if allow_model_override:
            model_default = agent_cfg.get("model", ModelConfig.MODELS[0])
            if model_default not in ModelConfig.MODELS:
                model_default = ModelConfig.MODELS[0]
            selected_model = st.selectbox(
                t("Model"),
                options=ModelConfig.MODELS,
                index=ModelConfig.MODELS.index(model_default),
                key=f"{tab_key}_model"
            )
        else:
            selected_model = agent_cfg.get("model", ModelConfig.MODELS[0])
            st.text_input(t("Model"), value=selected_model, disabled=True)

    with col2:
        max_tokens = st.number_input(
            t("Max Tokens"),
            min_value=1000,
            max_value=120000,
            value=agent_cfg.get("max_tokens", 12000),
            step=1000,
            key=f"{tab_key}_tokens"
        )

    with col3:
        temperature = st.number_input(
            t("Temperature"),
            min_value=0.0,
            max_value=1.0,
            value=float(agent_cfg.get("temperature", 0.2)),
            step=0.1,
            key=f"{tab_key}_temp"
        )

    # Prompt input
    prompt_key = f"{tab_key}_prompt"
    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = override_prompt or default_prompt

    user_prompt = st.text_area(
        "User Prompt",
        value=st.session_state[prompt_key],
        height=150,
        key=f"{prompt_key}_widget"
    )
    st.session_state[prompt_key] = user_prompt

    # Input document / context
    input_key = f"{tab_key}_input"
    if input_key not in st.session_state:
        st.session_state[input_key] = default_input_text

    clipboard = st.session_state.get("agent_clipboard", "")
    if clipboard:
        use_clip = st.checkbox(
            "Use Global Clipboard as starting input",
            key=f"{tab_key}_use_clip"
        )
    else:
        use_clip = False

    if clipboard and use_clip and not st.session_state.get(f"{tab_key}_clip_initialized", False):
        st.session_state[input_key] = clipboard
        st.session_state[f"{tab_key}_clip_initialized"] = True

    input_text = st.text_area(
        "Input Document/Context",
        value=st.session_state[input_key],
        height=300,
        key=f"{input_key}_widget"
    )
    st.session_state[input_key] = input_text

    # Run button
    if st.button(t("Run Agent"), key=f"{tab_key}_run", type="primary"):
        st.session_state[status_key] = "running"
        st.rerun()

    # Execute if running
    if status == "running":
        try:
            with st.spinner("Agent processing..."):
                system_prompt = agent_cfg.get("system_prompt", "")
                full_user_prompt = f"{user_prompt}\n\n{input_text}"

                output = call_llm(
                    model=selected_model,
                    system_prompt=system_prompt,
                    user_prompt=full_user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Store output
                output_key = f"{tab_key}_output"
                st.session_state[output_key] = output
                st.session_state[status_key] = "done"

                # Log event
                log_event(
                    tab=tab_label_for_history or tab_key,
                    agent=agent_id,
                    model=selected_model,
                    tokens_est=max_tokens
                )

                st.rerun()
        except Exception as e:
            st.session_state[status_key] = "error"
            st.error(f"Error: {str(e)}")

    # Display output
    output_key = f"{tab_key}_output"
    if output_key in st.session_state and st.session_state[output_key]:
        st.markdown("---")
        st.markdown("### Output")

        edited_output = st.text_area(
            "Edit output (text or markdown)",
            value=st.session_state[output_key],
            height=400,
            key=f"{output_key}_edit"
        )

        # Save edited version
        st.session_state[f"{output_key}_edited"] = edited_output

        # Preview tabs
        preview_tabs = st.tabs(["🧾 Raw Text", "📄 Markdown Preview"])
        with preview_tabs[0]:
            st.text_area(
                "Raw Text View (read-only)",
                value=edited_output,
                height=200,
                key=f"{output_key}_preview_raw",
                disabled=True
            )
        with preview_tabs[1]:
            st.markdown(edited_output)

        col_dl, col_clip = st.columns(2)
        with col_dl:
            st.download_button(
                "📥 Download Output",
                data=edited_output,
                file_name=f"{agent_id}_output.md",
                mime="text/markdown",
                key=f"{output_key}_download"
            )
        with col_clip:
            if st.button("📎 Send to Global Clipboard", key=f"{output_key}_clipboard"):
                st.session_state["agent_clipboard"] = edited_output
                st.success("Output sent to Global Clipboard. You can reuse it in other agents.")


def log_event(tab: str, agent: str, model: str, tokens_est: int):
    """Log execution event for analytics"""
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].append({
        "tab": tab,
        "agent": agent,
        "model": model,
        "tokens_est": tokens_est,
        "ts": datetime.utcnow().isoformat()
    })


# ============================================================================
# UI COMPONENTS
# ============================================================================

def apply_global_theme():
    """Apply light/dark theming plus painter style background"""
    theme = st.session_state.get("theme", "Dark")
    painter = st.session_state.get("painter_style", UIConfig.PAINTER_STYLES[0])
    painter_css = UIConfig.STYLE_CSS.get(painter, "")

    # Base theme (text colors etc.)
    if theme == "Dark":
        base_css = """
        <style>
        body { color: #F5F5F5; background-color: #0e1117; }
        .stApp { color: #F5F5F5; background-color: transparent; }
        </style>
        """
    else:
        base_css = """
        <style>
        body { color: #111827; background-color: #FFFFFF; }
        .stApp { color: #111827; background-color: transparent; }
        </style>
        """

    st.markdown(base_css, unsafe_allow_html=True)

    # Painter style layered on body
    if painter_css:
        st.markdown(
            f"""
            <style>
            body {{
                {painter_css}
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


def render_api_key_block(
    provider_label: str,
    env_var: str,
    session_key: str,
    placeholder: str,
    checkbox_suffix: str
):
    """API key UI: hide environment key, allow user input only when needed"""
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {}
    current = st.session_state["api_keys"].get(session_key, "")
    env_val = os.getenv(env_var)

    if env_val:
        # Allow optional override but NEVER show the env key
        use_custom = st.sidebar.checkbox(
            f"Use custom {provider_label} key",
            value=bool(current),
            key=f"{session_key}_use_custom_{checkbox_suffix}"
        )
        if use_custom:
            new_val = st.sidebar.text_input(
                f"{provider_label} API Key",
                type="password",
                value=current,
                placeholder=placeholder,
                key=f"{session_key}_input_{checkbox_suffix}"
            )
            st.session_state["api_keys"][session_key] = new_val
        else:
            st.sidebar.success(f"{provider_label}: using environment key")
            # Do not store the env key in session; call_llm falls back to os.getenv
            st.session_state["api_keys"][session_key] = ""
    else:
        # No env key: must input on web
        new_val = st.sidebar.text_input(
            f"{provider_label} API Key",
            type="password",
            value=current,
            placeholder=placeholder,
            key=f"{session_key}_input_{checkbox_suffix}"
        )
        st.session_state["api_keys"][session_key] = new_val


def render_sidebar():
    """Render global sidebar configuration"""
    st.sidebar.title("⚙️ Settings")

    # Language selector
    language = st.sidebar.selectbox(
        "Language / 語言",
        options=["English", "繁體中文"],
        index=0,
        key="language_selector"
    )
    st.session_state["language"] = language

    # Theme selector
    theme = st.sidebar.selectbox(
        "Theme",
        options=["Light", "Dark"],
        index=1,
        key="theme_selector"
    )
    st.session_state["theme"] = theme

    # Painter style selector + Jackpot
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Painter Style")

    col1, col2 = st.sidebar.columns([3, 2])
    with col1:
        painter = st.selectbox(
            "Style",
            options=UIConfig.PAINTER_STYLES,
            index=0,
            key="painter"
        )
    with col2:
        if st.button("🎰 Jackpot", key="painter_jackpot"):
            import random
            painter = random.choice(UIConfig.PAINTER_STYLES)
            st.session_state["painter"] = painter
            st.session_state["painter_style"] = painter
            st.experimental_rerun()

    st.session_state["painter_style"] = painter

    # API Keys
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 API Keys")

    render_api_key_block(
        provider_label="OpenAI",
        env_var="OPENAI_API_KEY",
        session_key="openai",
        placeholder="sk-...",
        checkbox_suffix="openai"
    )
    render_api_key_block(
        provider_label="Gemini",
        env_var="GEMINI_API_KEY",
        session_key="gemini",
        placeholder="",
        checkbox_suffix="gemini"
    )
    render_api_key_block(
        provider_label="Anthropic",
        env_var="ANTHROPIC_API_KEY",
        session_key="anthropic",
        placeholder="sk-ant-...",
        checkbox_suffix="anthropic"
    )
    render_api_key_block(
        provider_label="Grok",
        env_var="GROK_API_KEY",
        session_key="grok",
        placeholder="xai-...",
        checkbox_suffix="grok"
    )

    # Quick system status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 System Status")
    agents_cfg = load_agents_config()
    n_agents = len(agents_cfg.get("agents", {}))
    st.sidebar.markdown(f"- **Agents Loaded:** {n_agents}")
    providers_ready = []
    for prov, env_var in [
        ("OpenAI", "OPENAI_API_KEY"),
        ("Gemini", "GEMINI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Grok", "GROK_API_KEY"),
    ]:
        if os.getenv(env_var) or st.session_state["api_keys"].get(prov.lower(), ""):
            providers_ready.append(f"✅ {prov}")
        else:
            providers_ready.append(f"⚠️ {prov}")
    st.sidebar.markdown("- " + " | ".join(providers_ready))

    # Upload custom agents.yaml (quick override)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Custom Agents (Quick Load)")
    uploaded_agents = st.sidebar.file_uploader(
        "Upload agents.yaml",
        type=["yaml", "yml"],
        help="Override default agent configuration for this session"
    )

    if uploaded_agents:
        try:
            agents_cfg = yaml.safe_load(uploaded_agents) or {}
            if "agents" not in agents_cfg:
                agents_cfg["agents"] = {}
            st.session_state["agents_cfg"] = agents_cfg
            st.sidebar.success("✅ Custom agents loaded into session")
        except Exception as e:
            st.sidebar.error(f"Error loading agents: {str(e)}")


def render_dashboard():
    """Render analytics dashboard"""
    st.title(t("Dashboard"))

    history = st.session_state.get("history", [])

    if not history:
        st.info("No activity yet. Start using agents to see analytics.")
        return

    import pandas as pd

    df = pd.DataFrame(history)
    df["ts"] = pd.to_datetime(df["ts"])

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Runs", len(df))

    with col2:
        tabs_used = len(set(df["tab"]))
        st.metric("Tabs Used", tabs_used)

    with col3:
        total_tokens = int(df["tokens_est"].sum())
        st.metric("Est. Tokens", f"{total_tokens:,}")

    # Activity breakdown
    st.markdown("---")
    st.subheader("Activity Breakdown")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Runs by Tab**")
        tab_counts = df["tab"].value_counts()
        st.bar_chart(tab_counts)
    with col_b:
        st.markdown("**Runs by Model**")
        model_counts = df["model"].value_counts()
        st.bar_chart(model_counts)

    # Timeline
    st.markdown("---")
    st.subheader("Timeline (Daily Estimated Tokens)")
    df_time = df.set_index("ts").sort_index()
    daily_tokens = df_time["tokens_est"].resample("D").sum()
    st.area_chart(daily_tokens)

    # Recent activity table
    st.markdown("---")
    st.subheader("Recent Activity (Last 25 Runs)")
    st.dataframe(df.sort_values("ts", ascending=False).head(25), use_container_width=True)


def render_510k_intelligence_tab():
    """510(k) Intelligence tab"""
    st.title(t("510k_tab"))

    st.markdown("""
    Generate comprehensive device overview from FDA databases and public sources.
    """)

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        device_name = st.text_input("Device Name", key="510k_device_name")
        k_number = st.text_input("510(k) Number", key="510k_number")
    with col2:
        sponsor = st.text_input("Sponsor/Manufacturer", key="510k_sponsor")
        product_code = st.text_input("Product Code", key="510k_product_code")

    additional_context = st.text_area(
        "Additional Context",
        height=150,
        key="510k_context"
    )

    # Build prompt
    prompt = f"""
Device Name: {device_name}
510(k) Number: {k_number}
Sponsor: {sponsor}
Product Code: {product_code}

Additional Context:
{additional_context}
"""

    agent_run_ui(
        agent_id="fda_search_agent",
        tab_key="510k_intel",
        default_prompt="Generate comprehensive device overview with 5+ tables.",
        default_input_text=prompt,
        tab_label_for_history="510(k) Intelligence"
    )


def render_pdf_to_markdown_tab():
    """PDF to Markdown conversion tab"""
    st.title(t("pdf_tab"))

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start Page", min_value=1, value=1, key="pdf_start")
        with col2:
            end_page = st.number_input("End Page", min_value=1, value=10, key="pdf_end")

        if st.button("📄 Extract Text", key="pdf_extract"):
            with st.spinner("Extracting text from PDF..."):
                text = extract_pdf_pages_to_text(uploaded_file, start_page, end_page)
                st.session_state["pdf_raw_text"] = text
                st.success(f"✅ Extracted {len(text)} characters")

    # Show extracted text
    if "pdf_raw_text" in st.session_state:
        st.markdown("---")
        st.subheader("Extracted Text")
        st.text_area(
            "Raw Text",
            value=st.session_state["pdf_raw_text"],
            height=300,
            key="pdf_raw_display"
        )

        # Convert to markdown
        agent_run_ui(
            agent_id="pdf_to_markdown_agent",
            tab_key="pdf_to_md",
            default_prompt="Convert to clean markdown preserving structure.",
            default_input_text=st.session_state["pdf_raw_text"],
            tab_label_for_history="PDF to Markdown"
        )


def render_summary_entities_tab():
    """Summary & Entities extraction tab"""
    st.title(t("summary_tab"))

    # Option to pull from PDF tab
    use_pdf_output = st.checkbox("Use output from PDF → Markdown tab")
    if use_pdf_output:
        if "pdf_to_md_output_edited" in st.session_state:
            input_text = st.session_state["pdf_to_md_output_edited"]
        else:
            input_text = ""
            st.warning("No output available from PDF tab yet")
    else:
        input_text = ""

    agent_run_ui(
        agent_id="summary_entities_agent",
        tab_key="summary_entities",
        default_prompt="Generate 3000-4000 word summary with 20+ entity table.",
        default_input_text=input_text,
        tab_label_for_history="Summary & Entities"
    )


def render_comparator_tab():
    """Document comparison tab"""
    st.title(t("diff_tab"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Old Version")
        old_file = st.file_uploader("Upload Old PDF", type=["pdf"], key="diff_old")
        if old_file and st.button("Extract Old", key="extract_old"):
            text = extract_pdf_pages_to_text(old_file, 1, 9999)
            st.session_state["old_text"] = text
            st.success(f"✅ {len(text)} chars")

    with col2:
        st.subheader("New Version")
        new_file = st.file_uploader("Upload New PDF", type=["pdf"], key="diff_new")
        if new_file and st.button("Extract New", key="extract_new"):
            text = extract_pdf_pages_to_text(new_file, 1, 9999)
            st.session_state["new_text"] = text
            st.success(f"✅ {len(text)} chars")

    # Run comparison
    if "old_text" in st.session_state and "new_text" in st.session_state:
        combined_input = f"""
OLD VERSION:
{st.session_state['old_text']}

---

NEW VERSION:
{st.session_state['new_text']}
"""

        agent_run_ui(
            agent_id="diff_agent",
            tab_key="comparator",
            default_prompt="Identify 100+ substantive differences.",
            default_input_text=combined_input,
            tab_label_for_history="Comparator"
        )


def render_checklist_report_tab():
    """Checklist generation and review report tab"""
    st.title(t("checklist_tab"))

    st.markdown("### Stage 1: Generate Checklist from Guidance")

    guidance_input = st.text_area(
        "Paste Guidance Document or Upload",
        height=200,
        key="checklist_guidance"
    )

    uploaded_guidance = st.file_uploader(
        "Or upload guidance (PDF/TXT/MD)",
        type=["pdf", "txt", "md"],
        key="checklist_guidance_file"
    )

    if uploaded_guidance:
        if uploaded_guidance.name.endswith(".pdf"):
            guidance_input = extract_pdf_pages_to_text(uploaded_guidance, 1, 9999)
        else:
            guidance_input = uploaded_guidance.read().decode("utf-8")

    agent_run_ui(
        agent_id="guidance_to_checklist_converter",
        tab_key="checklist_gen",
        default_prompt="Generate structured checklist with 10+ domains.",
        default_input_text=guidance_input,
        tab_label_for_history="Checklist Generation"
    )

    st.markdown("---")
    st.markdown("### Stage 2: Generate Review Report")

    checklist_results = st.text_area(
        "Paste completed checklist results",
        height=300,
        key="checklist_results"
    )

    agent_run_ui(
        agent_id="review_memo_builder",
        tab_key="review_report",
        default_prompt="Compile comprehensive review memorandum.",
        default_input_text=checklist_results,
        tab_label_for_history="Review Report"
    )


def render_notes_magics_tab():
    """AI Note Keeper & Magics tab"""
    st.title(t("notes_tab"))

    st.markdown("### AI Note Keeper")

    notes_input = st.text_area(
        "Paste your note (text or markdown)",
        height=200,
        key="notes_input"
    )

    note_prompt = st.text_input(
        "Optional persistent prompt for this note (used by AI Magics)",
        key="notes_custom_prompt"
    )

    # Use note_keeper_agent to transform note into organized markdown
    agent_run_ui(
        agent_id="note_keeper_agent",
        tab_key="note_keeper",
        default_prompt=(
            "Transform this note into well-organized markdown with clear headings, "
            "bullet points, and actionable items. Preserve all important details."
        ),
        default_input_text=notes_input,
        tab_label_for_history="Note Keeper"
    )

    # Determine source text for magics
    organized_note = st.session_state.get("note_keeper_output_edited", "") or notes_input

    st.markdown("---")
    st.markdown("### AI Magics on This Note")

    magic_tab = st.selectbox(
        "Select Magic Tool",
        [
            "AI Formatting",
            "AI Keywords",
            "AI Action Items",
            "AI Concept Map",
            "AI Glossary",
            "AI Summarization",
        ],
        key="magic_selector"
    )

    use_organized = st.checkbox(
        "Use organized note from AI Note Keeper as input",
        value=True,
        key="magic_use_organized"
    )

    if use_organized and organized_note:
        magic_input_default = organized_note
    else:
        magic_input_default = ""

    magic_input = st.text_area(
        "Input for magic tool (leave blank if using organized note)",
        height=200,
        key="magic_input",
        value=magic_input_default
    )

    base_prompt = note_prompt.strip()
    if base_prompt:
        base_prompt += "\n\n"

    # Map to agent IDs
    magic_agents = {
        "AI Formatting": "magic_formatting_agent",
        "AI Keywords": "magic_keywords_agent",
        "AI Action Items": "magic_action_items_agent",
        "AI Concept Map": "magic_concept_map_agent",
        "AI Glossary": "magic_glossary_agent",
        "AI Summarization": "magic_summarization_agent",
    }

    override_prompt = None

    if magic_tab == "AI Keywords":
        keywords_str = st.text_input(
            "Keywords to highlight (comma-separated)",
            key="magic_keywords_list"
        )
        highlight_color = st.color_picker(
            "Highlight Color",
            "#FFEB3B",
            key="magic_keywords_color"
        )
        kw_list = [k.strip() for k in keywords_str.split(",") if k.strip()]
        kw_text = ", ".join(kw_list) if kw_list else "the most important technical and regulatory terms"
        override_prompt = (
            base_prompt
            + f"Highlight the following keywords in the note: {kw_text}.\n"
              f"Use markdown-compatible HTML spans with background color {highlight_color} "
              f"to highlight each occurrence. Preserve all other content and structure."
        )
    elif magic_tab == "AI Formatting":
        override_prompt = (
            base_prompt
            + "Reformat this note into clean, consistent markdown with proper headings, "
              "lists, tables where helpful, and fixed spacing/typos. Do not change the meaning."
        )
    elif magic_tab == "AI Action Items":
        override_prompt = (
            base_prompt
            + "Extract all action items, decisions, owners (if any), and due dates from the note. "
              "Output a markdown checklist plus a table of actions."
        )
    elif magic_tab == "AI Concept Map":
        override_prompt = (
            base_prompt
            + "Create a concept map of this note as markdown: list key concepts, their "
              "relationships, and optionally a mermaid diagram block."
        )
    elif magic_tab == "AI Glossary":
        override_prompt = (
            base_prompt
            + "Build a concise markdown glossary of important terms from the note "
              "(term, definition, category, relevance)."
        )
    elif magic_tab == "AI Summarization":
        override_prompt = (
            base_prompt
            + "Summarize the note in two parts: (1) a short bullet-point executive summary, "
              "(2) a structured, longer summary organized by headings."
        )

    agent_run_ui(
        agent_id=magic_agents[magic_tab],
        tab_key=f"magic_{magic_tab.lower().replace(' ', '_')}",
        default_prompt=override_prompt or f"Apply {magic_tab} transformation.",
        default_input_text=magic_input,
        tab_label_for_history=f"Magic: {magic_tab}",
        override_prompt=override_prompt
    )


def render_orchestration_tab():
    """FDA Reviewer Orchestration tab"""
    st.title(t("orch_tab"))

    st.markdown("""
    **Device-Specific Review Planning**: Generate comprehensive agent orchestration plan.
    """)

    st.markdown("### Step 1: Device Description")

    device_desc = st.text_area(
        "Enter device description (or upload PDF/DOCX)",
        height=200,
        key="orch_device_desc"
    )

    uploaded_device = st.file_uploader(
        "Or upload device description file",
        type=["pdf", "docx", "txt"],
        key="orch_device_file"
    )

    if uploaded_device:
        if uploaded_device.name.endswith(".pdf"):
            device_desc = extract_pdf_pages_to_text(uploaded_device, 1, 9999)
        elif uploaded_device.name.endswith(".docx"):
            device_desc = extract_docx_to_text(uploaded_device)
        else:
            device_desc = uploaded_device.read().decode("utf-8")

    st.markdown("### Step 2: Review Parameters")

    col1, col2 = st.columns(2)
    with col1:
        submission_type = st.selectbox(
            "Submission Type",
            ["Traditional 510(k)", "Special 510(k)", "Abbreviated 510(k)", "De Novo"],
            key="orch_sub_type"
        )

        predicates = st.text_input("Predicate Devices (comma-separated)", key="orch_predicates")

    with col2:
        clinical_data = st.selectbox(
            "Clinical Data Included?",
            ["Yes - Clinical study", "Yes - Literature", "No"],
            key="orch_clinical"
        )

        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Comprehensive"],
            value="Standard",
            key="orch_depth"
        )

    special_circumstances = st.text_area(
        "Special Circumstances (software, cybersecurity, combination product, etc.)",
        height=100,
        key="orch_special"
    )

    st.markdown("### Step 3: Generate Orchestration Plan")

    # Build orchestration prompt
    orch_prompt = f"""
Device Description:
{device_desc}

Submission Type: {submission_type}
Predicates: {predicates}
Clinical Data: {clinical_data}
Analysis Depth: {analysis_depth}
Special Circumstances: {special_circumstances}

Generate comprehensive review orchestration plan with:
1. Device classification analysis
2. Phase-based agent recommendations (Phases 1-4)
3. Execution sequence and parallel opportunities
4. Timeline estimates
5. Critical focus areas
6. Anticipated challenges
7. Ready-to-use agent commands
"""

    # Custom system prompt for orchestrator
    orch_system_prompt = (
        "You are an FDA regulatory review orchestration expert. Generate comprehensive, "
        "phase-based review plans using available agents catalog. Output must include "
        "detailed agent selection rationale and execution sequences."
    )

    # Model controls for orchestration
    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        orch_model = st.selectbox(
            "Model for Orchestration",
            ModelConfig.MODELS,
            index=0,
            key="orch_model"
        )
    with colm2:
        orch_tokens = st.number_input(
            "Max Tokens",
            min_value=4000,
            max_value=20000,
            value=16000,
            step=1000,
            key="orch_tokens"
        )
    with colm3:
        orch_temp = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            key="orch_temp"
        )

    if st.button("🎼 Generate Orchestration Plan", type="primary"):
        with st.spinner("Analyzing device and generating plan..."):
            try:
                plan = call_llm(
                    model=orch_model,
                    system_prompt=orch_system_prompt,
                    user_prompt=orch_prompt,
                    max_tokens=int(orch_tokens),
                    temperature=float(orch_temp)
                )
                st.session_state["orch_plan"] = plan
                st.success("✅ Plan generated")
                log_event("Orchestration", "orchestrator", orch_model, int(orch_tokens))
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display plan
    if "orch_plan" in st.session_state:
        st.markdown("---")
        st.markdown("### Orchestration Plan")

        edited_plan = st.text_area(
            "Review and edit plan",
            value=st.session_state["orch_plan"],
            height=600,
            key="orch_plan_edit"
        )

        st.download_button(
            "📥 Download Plan",
            data=edited_plan,
            file_name="orchestration_plan.md",
            mime="text/markdown"
        )


def render_dynamic_agents_tab():
    """Dynamic agent generation tab"""
    st.title(t("dynamic_tab"))

    st.markdown("""
    **AI-Driven Agent Creation**: Generate specialized review agents from FDA guidance documents.
    """)

    st.markdown("### Step 1: Upload Guidance Document")

    guidance_text = st.text_area(
        "Paste guidance text",
        height=200,
        key="dyn_guidance"
    )

    uploaded_guidance = st.file_uploader(
        "Or upload guidance (PDF/TXT/MD)",
        type=["pdf", "txt", "md"],
        key="dyn_guidance_file"
    )

    if uploaded_guidance:
        if uploaded_guidance.name.endswith(".pdf"):
            guidance_text = extract_pdf_pages_to_text(uploaded_guidance, 1, 9999)
        else:
            guidance_text = uploaded_guidance.read().decode("utf-8")

    st.markdown("### Step 2: Configuration")

    col1, col2 = st.columns(2)
    with col1:
        target_agent_count = st.slider(
            "Target Agent Count",
            min_value=3,
            max_value=8,
            value=5,
            key="dyn_count"
        )

    with col2:
        dyn_model = st.selectbox(
            "Model for Generation",
            ModelConfig.MODELS,
            index=0,
            key="dyn_model"
        )

    st.markdown("### Step 3: Generate Agents")

    dyn_system_prompt = (
        "You are an AI agent design expert for FDA regulatory review. Analyze the provided "
        "guidance document and existing agents catalog to generate 3-8 new, specialized, "
        "non-duplicative agent definitions in YAML format. Each agent must address specific "
        "guidance requirements not covered by existing agents."
    )

    if st.button("🤖 Generate Dynamic Agents", type="primary"):
        with st.spinner(f"Generating {target_agent_count} specialized agents..."):
            try:
                # Load current agents for context
                agents_cfg = load_agents_config()
                existing_agents_summary = "\n".join([
                    f"- {aid}: {acfg.get('name', aid)}"
                    for aid, acfg in agents_cfg.get("agents", {}).items()
                ])

                dyn_prompt = f"""
Guidance Document:
{guidance_text}

Existing Agents (do not duplicate):
{existing_agents_summary}

Generate {target_agent_count} new specialized agents in YAML format.
"""

                result = call_llm(
                    model=dyn_model,
                    system_prompt=dyn_system_prompt,
                    user_prompt=dyn_prompt,
                    max_tokens=20000,
                    temperature=0.4
                )

                st.session_state["dyn_agent_yaml"] = result
                st.success(f"✅ Generated {target_agent_count} agents")
                log_event("Dynamic Agents", "dynamic_generator", dyn_model, 20000)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display generated YAML
    if "dyn_agent_yaml" in st.session_state:
        st.markdown("---")
        st.markdown("### Generated Agents (YAML)")

        edited_yaml = st.text_area(
            "Review and edit YAML",
            value=st.session_state["dyn_agent_yaml"],
            height=600,
            key="dyn_yaml_edit"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📥 Download new_agents.yaml",
                data=edited_yaml,
                file_name="new_agents.yaml",
                mime="text/yaml"
            )
        with col2:
            if st.button("🔄 Merge with Current Agents"):
                try:
                    new_agents = yaml.safe_load(edited_yaml) or {}
                    current_agents = load_agents_config()
                    if "agents" not in current_agents:
                        current_agents["agents"] = {}
                    if "agents" in new_agents:
                        current_agents["agents"].update(new_agents.get("agents", {}))
                    else:
                        # assume entire YAML is just agents map
                        current_agents["agents"].update(new_agents)
                    st.session_state["agents_cfg"] = current_agents
                    st.success("✅ Agents merged! They are now available in this session.")
                except Exception as e:
                    st.error(f"Merge error: {str(e)}")
        with col3:
            if st.button("💾 Save merged agents.yaml to file"):
                try:
                    current_agents = load_agents_config()
                    with open("agents.yaml", "w", encoding="utf-8") as f:
                        yaml.safe_dump(current_agents, f, allow_unicode=True, sort_keys=False)
                    st.success("✅ Saved merged agents.yaml to file.")
                except Exception as e:
                    st.error(f"File save error: {str(e)}")


def render_config_files_tab():
    """Config & Files tab: modify/upload/download agents.yaml and SKILL.md"""
    st.title(t("config_files_tab"))

    # ---------------------- agents.yaml ----------------------
    st.markdown("## agents.yaml")

    agents_cfg = load_agents_config()
    agents_yaml_str = yaml.safe_dump(agents_cfg, allow_unicode=True, sort_keys=False)

    uploaded_agents_full = st.file_uploader(
        "Upload agents.yaml to replace current session configuration",
        type=["yaml", "yml"],
        key="config_agents_uploader"
    )

    if uploaded_agents_full:
        try:
            new_cfg = yaml.safe_load(uploaded_agents_full) or {}
            if "agents" not in new_cfg:
                new_cfg["agents"] = {}
            agents_yaml_str = yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False)
            st.session_state["agents_cfg"] = new_cfg
            st.success("✅ Uploaded agents.yaml loaded into session.")
        except Exception as e:
            st.error(f"Error parsing uploaded agents.yaml: {str(e)}")

    edited_agents_yaml = st.text_area(
        "Edit agents.yaml (session view)",
        value=agents_yaml_str,
        height=400,
        key="agents_yaml_editor"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Apply to Session", key="apply_agents_session"):
            try:
                new_cfg = yaml.safe_load(edited_agents_yaml) or {}
                if "agents" not in new_cfg:
                    new_cfg["agents"] = {}
                st.session_state["agents_cfg"] = new_cfg
                st.success("✅ Updated agents configuration in session.")
            except Exception as e:
                st.error(f"Error applying YAML: {str(e)}")
    with col2:
        if st.button("💾 Save to File (agents.yaml)", key="save_agents_file"):
            try:
                new_cfg = yaml.safe_load(edited_agents_yaml) or {}
                if "agents" not in new_cfg:
                    new_cfg["agents"] = {}
                with open("agents.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)
                st.session_state["agents_cfg"] = new_cfg
                st.success("✅ Saved agents.yaml to file.")
            except Exception as e:
                st.error(f"Error saving agents.yaml: {str(e)}")
    with col3:
        st.download_button(
            "📥 Download agents.yaml",
            data=edited_agents_yaml,
            file_name="agents.yaml",
            mime="text/yaml",
            key="download_agents_yaml"
        )

    st.markdown("---")

    # ---------------------- SKILL.md ----------------------
    st.markdown("## SKILL.md (Prompt / Skill Library)")

    skill_path = "SKILL.md"
    if "skill_md_content" not in st.session_state:
        if os.path.exists(skill_path):
            try:
                with open(skill_path, "r", encoding="utf-8") as f:
                    st.session_state["skill_md_content"] = f.read()
            except Exception:
                st.session_state["skill_md_content"] = "# SKILL Library\n\n"
        else:
            st.session_state["skill_md_content"] = "# SKILL Library\n\n"

    uploaded_skill = st.file_uploader(
        "Upload SKILL.md (or .md/.txt) to replace current content",
        type=["md", "txt"],
        key="skill_md_uploader"
    )

    if uploaded_skill:
        try:
            st.session_state["skill_md_content"] = uploaded_skill.read().decode("utf-8")
            st.success("✅ Uploaded SKILL.md loaded into session.")
        except Exception as e:
            st.error(f"Error reading uploaded SKILL.md: {str(e)}")

    skill_content = st.text_area(
        "Edit SKILL.md",
        value=st.session_state["skill_md_content"],
        height=400,
        key="skill_md_editor"
    )
    st.session_state["skill_md_content"] = skill_content

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("💾 Save SKILL.md to File", key="save_skill_file"):
            try:
                with open(skill_path, "w", encoding="utf-8") as f:
                    f.write(skill_content)
                st.success("✅ Saved SKILL.md to file.")
            except Exception as e:
                st.error(f"Error saving SKILL.md: {str(e)}")
    with col_s2:
        st.download_button(
            "📥 Download SKILL.md",
            data=skill_content,
            file_name="SKILL.md",
            mime="text/markdown",
            key="download_skill_md"
        )
    with col_s3:
        if st.button("🔄 Reload SKILL.md from File", key="reload_skill_file"):
            try:
                if os.path.exists(skill_path):
                    with open(skill_path, "r", encoding="utf-8") as f:
                        st.session_state["skill_md_content"] = f.read()
                    st.success("✅ Reloaded SKILL.md from file.")
                else:
                    st.warning("SKILL.md file not found on disk.")
            except Exception as e:
                st.error(f"Error reloading SKILL.md: {str(e)}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""

    # Page config
    st.set_page_config(
        page_title="FDA 510(k) Agentic AI Review System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {}

    # Render sidebar & theme
    render_sidebar()
    apply_global_theme()

    # Main title
    st.title("🏥 FDA 510(k) Agentic AI Review System")
    st.caption("Multi-Agent AI for Comprehensive Regulatory Review | Version 2.1")

    # Tab navigation
    tabs = st.tabs([
        t("Dashboard"),
        t("510k_tab"),
        t("pdf_tab"),
        t("summary_tab"),
        t("diff_tab"),
        t("checklist_tab"),
        t("notes_tab"),
        t("orch_tab"),
        t("dynamic_tab"),
        t("config_files_tab"),
    ])

    with tabs[0]:
        render_dashboard()

    with tabs[1]:
        render_510k_intelligence_tab()

    with tabs[2]:
        render_pdf_to_markdown_tab()

    with tabs[3]:
        render_summary_entities_tab()

    with tabs[4]:
        render_comparator_tab()

    with tabs[5]:
        render_checklist_report_tab()

    with tabs[6]:
        render_notes_magics_tab()

    with tabs[7]:
        render_orchestration_tab()

    with tabs[8]:
        render_dynamic_agents_tab()

    with tabs[9]:
        render_config_files_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>FDA 510(k) Agentic AI Review System | Powered by Multi-LLM Architecture</p>
            <p>Supporting: OpenAI GPT-4 • Google Gemini • Anthropic Claude • xAI Grok</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
