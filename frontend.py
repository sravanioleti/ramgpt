from __future__ import annotations

import os
import requests
import streamlit as st

API = os.getenv("RAMGPT_API", "http://localhost:8000")
st.set_page_config(page_title="RamGPT | Ticket Resolution", page_icon="🧭", layout="wide")


def api(method: str, path: str, payload: dict | None = None) -> dict | list:
    response = requests.request(method, f"{API}{path}", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def unique_messages(messages: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for message in messages:
        if cleaned and cleaned[-1]["role"] == message["role"] and cleaned[-1]["content"] == message["content"]:
            continue
        cleaned.append(message)
    return cleaned


try:
    health = api("GET", "/api/health")
except requests.RequestException:
    st.error("Backend is offline. Start the application with `python run.py`.")
    st.stop()

if "case" not in st.session_state:
    st.session_state["case"] = None

with st.sidebar:
    st.title("🧭 RamGPT")
    if st.button("＋ New ticket", use_container_width=True, type="primary"):
        st.session_state["case"] = None
        st.session_state["analysis"] = None
        st.rerun()
    st.divider()
    st.caption(f"{health['historical_tickets']:,} historical tickets")
    st.caption(f"Retrieval: {health.get('retrieval', 'hybrid search')}")
    st.caption("Human approval required")

st.markdown("## RamGPT")
st.caption("A conversational, human-approved workspace for ServiceNow ticket resolution")

case = st.session_state["case"]
if case is None:
    st.info("Start a new ticket conversation below.")
    with st.form("new_ticket"):
        initial = st.text_area("How can RamGPT help?", placeholder="VPN is not connecting. I see an error when I sign in.", height=120)
        start = st.form_submit_button("Create ticket", type="primary")
    if start and len(initial.strip()) >= 3:
        first_line = initial.strip().splitlines()[0][:240]
        try:
            created = api("POST", "/api/tickets/analyze", {"short_description": first_line, "description": initial.strip()})
        except requests.RequestException as error:
            st.error(f"RamGPT could not analyze the ticket yet: {error}. Please try again in a moment.")
            st.stop()
        st.session_state["case"] = api("GET", f"/api/tickets/{created['ticket_id']}")
        st.session_state["analysis"] = created
        st.rerun()
    st.stop()

ticket = case["ticket"]
analysis = st.session_state.get("analysis")
if not analysis:
    analysis = api("GET", f"/api/tickets/{ticket['id']}/analysis")
    st.session_state["analysis"] = analysis

left, right = st.columns([1.45, 1], gap="large")
with left:
    st.subheader(ticket["short_description"])
    st.caption(f"{ticket['id']} · Status: {ticket['status'].replace('_', ' ').title()}")
    st.caption("Ask naturally. I can explain confidence, compare evidence, collect missing details, or summarize the handoff.")
    for message in case["messages"]:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask RamGPT: Why is confidence low? Show the most similar ticket...")
    if prompt:
        try:
            result = api("POST", f"/api/tickets/{ticket['id']}/chat", {"message": prompt})
        except requests.RequestException as error:
            st.error(f"RamGPT could not answer this message: {error}")
        else:
            result["analysis"]["context"] = result["context"]
            st.session_state["analysis"] = result["analysis"]
            updated_case = dict(case)
            updated_case["messages"] = unique_messages(
                list(case["messages"])
                + [
                    {"role": "user", "content": result["user_message"]},
                    {"role": "assistant", "content": result["assistant_message"]},
                ]
            )
            st.session_state["case"] = updated_case
            st.rerun()

    st.divider()
    st.subheader("Human decision")
    reviewer = st.text_input("Reviewer", value="Support Lead")
    reason = st.text_input("Decision reason", placeholder="Validate evidence and proposed team")
    approve, reject = st.columns(2)
    if approve.button("Approve recommendation", use_container_width=True, type="primary"):
        if reason.strip():
            api("POST", f"/api/tickets/{ticket['id']}/decision", {"action": "approve", "reviewer": reviewer, "comment": reason})
            st.session_state["case"] = api("GET", f"/api/tickets/{ticket['id']}")
            st.success("Approved and recorded in the audit trail.")
            st.rerun()
        st.warning("Add a decision reason.")
    if reject.button("Reject / escalate", use_container_width=True):
        if reason.strip():
            api("POST", f"/api/tickets/{ticket['id']}/decision", {"action": "reject", "reviewer": reviewer, "comment": reason})
            st.session_state["case"] = api("GET", f"/api/tickets/{ticket['id']}")
            st.warning("Rejected and escalated to human triage.")
            st.rerun()
        st.warning("Add a rejection reason.")

with right:
    st.subheader("Case context")
    context = analysis.get("context", {})
    with st.expander("Current case memory", expanded=True):
        st.markdown(f"**Ticket:** {ticket['id']}")
        st.markdown(f"**Issue:** {context.get('issue', ticket['short_description'])}")
        st.markdown(f"**Error:** {context.get('error') or 'Not provided'}")
        st.markdown(f"**Affected users:** {context.get('affected_users', 'Unknown')}")
        st.markdown(f"**Internet:** {context.get('internet') or 'Unknown'}")
        st.markdown("**Troubleshooting:** " + (", ".join(context.get("troubleshooting", [])) or "Not provided"))
    st.metric("Confidence", f"{analysis['confidence']:.0%}")
    st.caption(f"{analysis['confidence_band'].upper()} CONFIDENCE · {analysis['recommendation'].replace('_', ' ').title()}")
    components = analysis["confidence_components"]
    with st.expander("Why this confidence?", expanded=True):
        st.write(f"Retrieval similarity: {components['retrieval_similarity']:.0%}")
        st.write(f"Historical team agreement: {components['historical_agreement']:.0%}")
        st.write(f"Evidence quality: {components['evidence_quality']:.0%}")
        st.caption(analysis["decision_rationale"])
    if analysis["missing_information"]:
        with st.expander("Information still needed", expanded=True):
            for question in analysis["questions_for_requester"]:
                st.write(f"• {question}")
    st.subheader("Similarity evidence")
    for match in analysis["similar_tickets"][:3]:
        with st.expander(f"{match['ticket_id']} · Semantic similarity {match.get('semantic_similarity', match['similarity']):.2f}"):
            st.caption(f"{match['category']} · {match['assignment_group']}")
            st.write(match["resolution"])
    if analysis["proposed_resolution"]:
        st.subheader("Proposed resolution")
        st.info(analysis["proposed_resolution"])
    else:
        st.info("Low-confidence mode: no specific resolution is proposed until a human validates more evidence.")

st.divider()
st.caption("ServiceNow remains the system of record. Conversation history is a decision-support audit trail, not an autonomous ticket update.")
