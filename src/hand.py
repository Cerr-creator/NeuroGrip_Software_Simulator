import streamlit as st

HAND_INFO = {
    "Rest":      ("🫳", "Rest / relaxed"),
    "Open Hand": ("🖐️", "Open position"),
    "Fist":      ("✊", "Fist / closed"),
    "Pinch":     ("🤏", "Pinch position"),
    "Point":     ("☝️", "Point position"),
}

def render_hand(gesture):
    emoji, label = HAND_INFO.get(gesture, ("🖐️", gesture))

    st.markdown(
        f"""
        <div style="
            min-height:255px;
            border-radius:18px;
            border:1px solid #334155;
            background:linear-gradient(145deg,#0f172a,#111827);
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;
        ">
            <div style="font-size:7rem; line-height:1.1;">{emoji}</div>
            <div style="font-size:1.5rem;font-weight:700;margin-top:.5rem;">
                {label}
            </div>
            <div style="color:#94a3b8;margin-top:.3rem;">
                AI command: {gesture}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
