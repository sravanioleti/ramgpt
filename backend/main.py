from __future__ import annotations

import csv
import logging
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
logger = logging.getLogger("ramgpt")

try:
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    BM25Okapi = None
    SentenceTransformer = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


DATASET = Path(os.getenv("RAMGPT_DATASET", ROOT / "it_ticket_dataset.csv"))
DB_PATH = Path(os.getenv("RAMGPT_DB", ROOT / "ramgpt.db"))
CACHE_DIR = Path(os.getenv("RAMGPT_CACHE", ROOT / ".cache"))
FAISS_CACHE = CACHE_DIR / "historical_tickets.faiss"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with closing(db()) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY, short_description TEXT NOT NULL,
                description TEXT NOT NULL, category TEXT, assignment_group TEXT,
                status TEXT NOT NULL, confidence REAL, recommendation TEXT,
                proposed_resolution TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, action TEXT NOT NULL,
                reviewer TEXT NOT NULL, comment TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, rating INTEGER NOT NULL,
                corrected_group TEXT, corrected_resolution TEXT, comment TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, created_at TEXT NOT NULL
            );
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(feedback)")}
        if "final_resolution" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN final_resolution TEXT")
        if "recommendation_snapshot" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN recommendation_snapshot TEXT")
        if "confidence" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN confidence REAL")
        if "retrieved_ticket_ids" not in columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN retrieved_ticket_ids TEXT")
        connection.commit()


def load_history() -> list[dict[str, str]]:
    with DATASET.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


HISTORY = load_history()
TEXTS = [
    f"{row['short_description']} {row['description']} {row['category']} {row['assignment_group']}"
    for row in HISTORY
]
VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=120_000)
MATRIX = VECTORIZER.fit_transform(TEXTS)
TOKENS = [text.lower().split() for text in TEXTS]
BM25 = BM25Okapi(TOKENS) if BM25Okapi else None
EMBEDDER = None
FAISS_INDEX = None


def semantic_index():
    global EMBEDDER, FAISS_INDEX
    if os.getenv("RAMGPT_DISABLE_SEMANTIC") == "1":
        return None
    if SentenceTransformer is None or faiss is None:
        return None
    if FAISS_INDEX is None:
        EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        if FAISS_CACHE.exists():
            FAISS_INDEX = faiss.read_index(str(FAISS_CACHE))
        else:
            embeddings = EMBEDDER.encode(
                TEXTS,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=128,
            )
            FAISS_INDEX = faiss.IndexFlatIP(embeddings.shape[1])
            FAISS_INDEX.add(embeddings)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            temporary_cache = FAISS_CACHE.with_suffix(".tmp")
            faiss.write_index(FAISS_INDEX, str(temporary_cache))
            os.replace(temporary_cache, FAISS_CACHE)
    return FAISS_INDEX

app = FastAPI(title="RamGPT Ticket Resolution API", version="1.0.0")
init_db()


@app.on_event("startup")
def warm_retrieval_index() -> None:
    if os.getenv("RAMGPT_DISABLE_SEMANTIC") != "1":
        semantic_index()


class TicketRequest(BaseModel):
    short_description: str = Field(min_length=3, max_length=240)
    description: str = Field(min_length=3, max_length=4000)


class DecisionRequest(BaseModel):
    action: Literal["approve", "reject", "edit"]
    reviewer: str = Field(min_length=2, max_length=120)
    comment: str = Field(default="", max_length=1000)
    assignment_group: str | None = None
    resolution: str | None = None


class FeedbackRequest(BaseModel):
    rating: Literal[1, 0, -1]
    corrected_group: str | None = None
    corrected_resolution: str | None = None
    final_resolution: str | None = None
    retrieved_ticket_ids: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    comment: str = Field(default="", max_length=1000)


def retrieve(
    ticket: TicketRequest,
    limit: int = 5,
    exclude_indices: set[int] | None = None,
) -> list[dict]:
    query_text = f"{ticket.short_description} {ticket.description}"
    lexical = cosine_similarity(VECTORIZER.transform([query_text]), MATRIX).ravel()
    semantic = {}
    index = semantic_index()
    if index is not None:
        vector = EMBEDDER.encode([query_text], normalize_embeddings=True)
        distances, indexes = index.search(vector, min(20, len(HISTORY)))
        semantic = {int(item): float(score) for item, score in zip(indexes[0], distances[0])}
    bm25_scores = BM25.get_scores(query_text.lower().split()) if BM25 else [0.0] * len(HISTORY)
    max_bm25 = max(bm25_scores) or 1.0
    scores = {
        index: (0.55 * semantic.get(index, float(lexical[index])) +
                0.25 * float(lexical[index]) +
                0.20 * (float(bm25_scores[index]) / max_bm25))
        for index in range(len(HISTORY))
    }
    for index in exclude_indices or set():
        scores[index] = -1.0
    indexes = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [
        {**HISTORY[index], "similarity": round(float(scores[index]), 4),
         "semantic_similarity": round(float(semantic.get(index, lexical[index])), 4)}
        for index in indexes
        if scores[index] > 0
    ]


def evidence_quality(ticket: TicketRequest) -> tuple[float, list[str], list[str]]:
    text = f"{ticket.short_description} {ticket.description}".lower()
    signals = [
        len(ticket.description.split()) >= 12,
        any(word in text for word in ("error", "code", "message", "exception")),
        any(word in text for word in ("started", "since", "today", "yesterday", "noticed")),
        any(word in text for word in ("impact", "blocked", "cannot", "unable", "affect")),
    ]
    quality = sum(signals) / len(signals)
    missing: list[str] = []
    questions: list[str] = []
    if not signals[1]:
        missing.append("No error message or error code provided")
        questions.append("What error message or code do you see?")
    if not signals[2]:
        missing.append("When the issue started is unknown")
        questions.append("When did the issue start, and is it happening now?")
    if not signals[3]:
        missing.append("Business impact is not described")
        questions.append("What work is blocked or affected?")
    if not any(word in text for word in ("user", "users", "employee", "team", "everyone")):
        missing.append("Scope (one user or multiple users) is unknown")
        questions.append("Are other employees experiencing the same issue?")
    return quality, missing, questions


def decide(ticket: TicketRequest, matches: list[dict]) -> dict:
    if not matches:
        return {
            "confidence": 0.0, "band": "low", "recommendation": "human_review",
            "components": {"retrieval_similarity": 0.0, "historical_agreement": 0.0, "evidence_quality": 0.0},
            "missing_information": ["No historical ticket had measurable similarity"],
            "questions": ["Can you provide the affected service, symptoms, timing, and business impact?"],
            "rationale": "No historical evidence was strong enough to support a recommendation.",
        }
    top = matches[0]["similarity"]
    same_group = sum(match["assignment_group"] == matches[0]["assignment_group"] for match in matches)
    agreement = same_group / len(matches)
    quality, missing, questions = evidence_quality(ticket)
    # This is deliberately simple and inspectable: similarity is evidence, not confidence.
    confidence = min(0.99, (top * 0.50) + (agreement * 0.30) + (quality * 0.20))
    if confidence >= 0.80:
        band, recommendation = "high", "ready_for_approval"
    elif confidence >= 0.60:
        band, recommendation = "medium", "assisted_review"
    else:
        band, recommendation = "low", "human_review"
    return {
        "confidence": round(confidence, 4), "band": band, "recommendation": recommendation,
        "components": {
            "retrieval_similarity": top,
            "historical_agreement": round(agreement, 4),
            "evidence_quality": round(quality, 4),
        },
        "missing_information": missing, "questions": questions,
        "rationale": "Confidence combines retrieval similarity (50%), agreement on the suggested team (30%), and evidence quality in the new ticket (20%).",
    }


def context_from_conversation(messages: list[dict]) -> dict:
    text = " ".join(message["content"] for message in messages if message["role"] == "user").lower()
    context = {
        "issue": messages[0]["content"] if messages else "Not provided",
        "error": None,
        "affected_users": "Multiple or unknown",
        "troubleshooting": [],
        "internet": None,
    }
    if any(marker in text for marker in ("error 691", "error code 691", "691")):
        context["error"] = "691"
    if any(phrase in text for phrase in ("only me", "only one", "just me", "single user")):
        context["affected_users"] = "1 (single user)"
    if any(phrase in text for phrase in ("everyone", "all users", "multiple users", "team affected")):
        context["affected_users"] = "Multiple users"
    if "internet" in text and any(word in text for word in ("working", "works", "available")):
        context["internet"] = "Working"
    if "restart" in text or "reboot" in text:
        context["troubleshooting"].append("Restart/reboot attempted")
    if "retry" in text or "retried" in text:
        context["troubleshooting"].append("Connection retried")
    return context


def deduplicate_messages(messages: list[dict]) -> list[dict]:
    """Remove accidental repeated turns from earlier Streamlit reruns."""
    cleaned: list[dict] = []
    for message in messages:
        if cleaned and cleaned[-1]["role"] == message["role"] and cleaned[-1]["content"] == message["content"]:
            continue
        cleaned.append(message)
    return cleaned


def initial_message_content(ticket: TicketRequest) -> str:
    description = ticket.description.strip()
    short = ticket.short_description.strip()
    if description.lower().startswith(short.lower()):
        return description
    return f"{short}\n{description}"


def llm_response(user_text: str, messages: list[dict], analysis: dict, context: dict) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or genai is None:
        return None
    evidence = "\n".join(
        f"{item['ticket_id']} | similarity={item['similarity']:.2f} | "
        f"team={item['assignment_group']} | resolution={item['resolution']}"
        for item in analysis["similar_tickets"][:5]
    )
    transcript = "\n".join(f"{item['role']}: {item['content']}" for item in messages[-12:])
    prompt = f"""You are RamGPT, a careful IT service-desk copilot.
Answer the user's latest message conversationally and directly.
Use only the ticket conversation and historical evidence below. Never claim a historical
resolution is the root cause of the current issue. If evidence is weak, say so and ask
the most useful next question. If the user asks why, explain the confidence components.
If the user asks for historical tickets, cite their IDs and similarity scores.
Do not approve, close, or reassign tickets; a human reviewer does that.

Current structured context: {context}
Confidence analysis: {analysis['confidence']:.0%} ({analysis['confidence_band']}); {analysis['decision_rationale']}
Missing information: {analysis['missing_information']}
Historical evidence:
{evidence or "No sufficiently similar historical tickets."}
Conversation:
{transcript}
Latest user message: {user_text}
"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        return text or None
    except Exception as error:
        logger.warning("Gemini response failed; using grounded fallback: %s", error)
        return None


def fallback_response(user_text: str, analysis: dict) -> str:
    lowered = user_text.lower()
    if "why" in lowered or "confidence" in lowered:
        components = analysis["confidence_components"]
        return (
            f"Your current confidence is {analysis['confidence']:.0%} ({analysis['confidence_band']}). "
            f"That comes from semantic/lexical retrieval {components['retrieval_similarity']:.0%}, "
            f"historical team agreement {components['historical_agreement']:.0%}, and "
            f"evidence quality {components['evidence_quality']:.0%}. "
            "Similarity is evidence of related wording, not proof that the historical root cause applies."
        )
    if any(word in lowered for word in ("historical", "previous", "similar", "compare")):
        if not analysis["similar_tickets"]:
            return "I could not find a sufficiently similar historical ticket."
        return "Here are the strongest historical references:\n\n" + "\n\n".join(
            f"**{item['ticket_id']}** — similarity {item['similarity']:.2f}\n"
            f"Team: {item['assignment_group']}\nResolution: {item['resolution']}"
            for item in analysis["similar_tickets"][:3]
        )
    if analysis["missing_information"] and not analysis["proposed_resolution"]:
        team = analysis["proposed_assignment_group"] or "the service-desk triage team"
        questions = "\n".join(f"- {question}" for question in analysis["questions_for_requester"])
        return (
            f"I understand the reported problem and found related historical tickets, but I am "
            f"only **{analysis['confidence']:.0%} confident**, so I do not want to invent a root cause "
            f"or tell you to apply the wrong fix.\n\n"
            f"**What I can say:** this should first be reviewed by **{team}**. "
            "They can validate the service, authentication, and endpoint health before changing anything.\n\n"
            f"**What I need next:**\n{questions}\n\n"
            "**Recommended next step:** collect those details and consult the suggested team. "
            "A support engineer from that team should resolve the ticket after human review."
        )
    return (
        f"I found {len(analysis['similar_tickets'])} relevant historical tickets. "
        f"My suggested team is **{analysis['proposed_assignment_group'] or 'Human triage'}** "
        f"with **{analysis['confidence']:.0%} confidence**. "
        f"The historical evidence suggests this area, but it does not prove the current root cause. "
        "I can explain the evidence, compare previous tickets, or collect missing details. "
        "A human reviewer must approve the next action."
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok", "historical_tickets": len(HISTORY),
        "retrieval": "hybrid FAISS + BM25 + TF-IDF" if FAISS_INDEX is not None else "BM25 + TF-IDF",
        "llm": bool((os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) and genai),
    }


@app.get("/api/evaluation")
def evaluation() -> dict:
    sample = HISTORY[:50]
    top1 = 0
    top3 = 0
    for index, row in enumerate(sample):
        ticket = TicketRequest(
            short_description=row["short_description"],
            description=row["description"],
        )
        matches = retrieve(ticket, limit=3, exclude_indices={index})
        groups = [match["assignment_group"] for match in matches]
        top1 += int(groups and groups[0] == row["assignment_group"])
        top3 += int(row["assignment_group"] in groups)
    with closing(db()) as connection:
        feedback_rows = connection.execute("SELECT rating FROM feedback").fetchall()
    approved = sum(row["rating"] == 1 for row in feedback_rows)
    corrected = sum(row["rating"] == 0 for row in feedback_rows)
    rejected = sum(row["rating"] == -1 for row in feedback_rows)
    return {
        "sample_size": len(sample),
        "retrieval_top1_team_accuracy": round(top1 / len(sample), 3),
        "retrieval_top3_team_coverage": round(top3 / len(sample), 3),
        "feedback": {"approved": approved, "corrected": corrected, "rejected": rejected},
        "note": "Metrics use the production hybrid FAISS + BM25 + TF-IDF retriever. Each sampled ticket is excluded from its own nearest-neighbor search; resolution correctness requires a human-labeled benchmark.",
    }


@app.post("/api/tickets/analyze")
def analyze(ticket: TicketRequest, persist: bool = True) -> dict:
    matches = retrieve(ticket)
    decision = decide(ticket, matches)
    best = matches[0] if matches else None
    safe_to_propose = decision["band"] in ("high", "medium") and best is not None
    ticket_id = f"RAM-{uuid.uuid4().hex[:8].upper()}"
    timestamp = now()
    if not persist:
        return {
            "ticket_id": ticket_id,
            "confidence": decision["confidence"],
            "confidence_band": decision["band"],
            "recommendation": decision["recommendation"],
            "confidence_components": decision["components"],
            "decision_rationale": decision["rationale"],
            "missing_information": decision["missing_information"],
            "questions_for_requester": decision["questions"],
            "proposed_category": best["category"] if best else None,
            "proposed_assignment_group": best["assignment_group"] if best else None,
            "proposed_resolution": best["resolution"] if safe_to_propose else None,
            "suggested_investigation": [
                "Verify authentication and access prerequisites",
                "Check the affected service or endpoint health",
                "Confirm whether the issue affects other users",
            ],
            "similar_tickets": matches,
        }
    with closing(db()) as connection:
        connection.execute(
            """INSERT INTO tickets
            (id, short_description, description, category, assignment_group, status,
             confidence, recommendation, proposed_resolution, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticket_id, ticket.short_description, ticket.description,
                best["category"] if best else "Needs classification",
                best["assignment_group"] if best else "Human triage",
                "pending_approval", decision["confidence"], decision["recommendation"],
                best["resolution"] if safe_to_propose else None, timestamp, timestamp,
            ),
        )
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), ticket_id, "user", initial_message_content(ticket), timestamp),
        )
        initial_analysis = analyze(ticket, persist=False)
        initial_analysis["ticket_id"] = ticket_id
        initial_messages = [{
            "role": "user",
            "content": initial_message_content(ticket),
        }]
        initial_context = context_from_conversation(initial_messages)
        assistant_response = llm_response(
            ticket.description,
            initial_messages,
            initial_analysis,
            initial_context,
        ) or fallback_response(ticket.description, initial_analysis)
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), ticket_id, "assistant", assistant_response, now()),
        )
        connection.commit()
    return {
        "ticket_id": ticket_id,
        "confidence": decision["confidence"],
        "confidence_band": decision["band"],
        "recommendation": decision["recommendation"],
        "confidence_components": decision["components"],
        "decision_rationale": decision["rationale"],
        "missing_information": decision["missing_information"],
        "questions_for_requester": decision["questions"],
        "proposed_category": best["category"] if best else None,
        "proposed_assignment_group": best["assignment_group"] if best else None,
        "proposed_resolution": best["resolution"] if safe_to_propose else None,
        "suggested_investigation": [
            "Verify authentication and access prerequisites",
            "Check the affected service or endpoint health",
            "Confirm whether the issue affects other users",
        ],
        "similar_tickets": matches,
        "handoff": {
            "team": best["assignment_group"] if best else "Human triage",
            "reason": ticket.short_description,
            "reported": ticket.description,
            "historical_ticket_ids": [match["ticket_id"] for match in matches[:3]],
            "limitation": "Historical evidence does not establish root cause; a reviewer must validate the next action.",
        },
        "guardrail": "No ticket state changes until a human approves this recommendation.",
    }


@app.get("/api/tickets")
def list_tickets() -> list[dict]:
    with closing(db()) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM tickets ORDER BY created_at DESC LIMIT 50"
        ).fetchall()]


@app.get("/api/tickets/{ticket_id}")
def get_ticket(ticket_id: str) -> dict:
    with closing(db()) as connection:
        ticket = connection.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        messages = connection.execute(
            "SELECT role, content, created_at FROM messages WHERE ticket_id = ? ORDER BY created_at",
            (ticket_id,),
        ).fetchall()
    return {"ticket": dict(ticket), "messages": deduplicate_messages([dict(message) for message in messages])}


@app.post("/api/tickets/{ticket_id}/chat")
def chat(ticket_id: str, request: ChatRequest) -> dict:
    with closing(db()) as connection:
        ticket = connection.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), ticket_id, "user", request.message, now()),
        )
        rows = connection.execute(
            "SELECT role, content FROM messages WHERE ticket_id = ? ORDER BY created_at",
            (ticket_id,),
        ).fetchall()
        connection.commit()
    messages = deduplicate_messages([dict(row) for row in rows])
    context = context_from_conversation(messages)
    combined = " ".join(message["content"] for message in messages if message["role"] == "user")
    analysis = analyze(TicketRequest(short_description=ticket["short_description"], description=combined), persist=False)
    response = llm_response(request.message, messages, analysis, context)
    if response is None:
        response = fallback_response(request.message, analysis)
    with closing(db()) as connection:
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), ticket_id, "assistant", response, now()),
        )
        connection.execute(
            "UPDATE tickets SET confidence=?, recommendation=?, updated_at=? WHERE id=?",
            (analysis["confidence"], analysis["recommendation"], now(), ticket_id),
        )
        connection.commit()
    return {
        "response": response,
        "context": context,
        "analysis": analysis,
        "user_message": request.message,
        "assistant_message": response,
    }


@app.get("/api/tickets/{ticket_id}/analysis")
def ticket_analysis(ticket_id: str) -> dict:
    with closing(db()) as connection:
        ticket = connection.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        messages = connection.execute(
            "SELECT role, content FROM messages WHERE ticket_id = ? ORDER BY created_at",
            (ticket_id,),
        ).fetchall()
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    messages = deduplicate_messages([dict(row) for row in messages])
    combined = " ".join(row["content"] for row in messages if row["role"] == "user")
    analysis = analyze(
        TicketRequest(short_description=ticket["short_description"], description=combined),
        persist=False,
    )
    analysis["context"] = context_from_conversation([dict(row) for row in messages])
    return analysis


@app.post("/api/tickets/{ticket_id}/decision")
def decision(ticket_id: str, request: DecisionRequest) -> dict:
    with closing(db()) as connection:
        ticket = connection.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        status = "approved" if request.action in ("approve", "edit") else "rejected"
        group = request.assignment_group or ticket["assignment_group"]
        resolution = request.resolution or ticket["proposed_resolution"]
        connection.execute(
            "UPDATE tickets SET status=?, assignment_group=?, proposed_resolution=?, updated_at=? WHERE id=?",
            (status, group, resolution, now(), ticket_id),
        )
        connection.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), ticket_id, request.action, request.reviewer, request.comment, now()),
        )
        connection.commit()
    return {
        "ticket_id": ticket_id, "status": status, "assignment_group": group,
        "resolution": resolution, "audit": {
            "action": request.action, "reviewer": request.reviewer,
            "comment": request.comment, "timestamp": now(),
        },
    }


@app.post("/api/tickets/{ticket_id}/feedback")
def feedback(ticket_id: str, request: FeedbackRequest) -> dict:
    with closing(db()) as connection:
        ticket = connection.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        connection.execute(
            """INSERT INTO feedback
            (id, ticket_id, rating, corrected_group, corrected_resolution, comment,
             created_at, final_resolution, recommendation_snapshot, confidence, retrieved_ticket_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), ticket_id, request.rating, request.corrected_group,
             request.corrected_resolution, request.comment, now(),
             request.final_resolution, ticket["proposed_resolution"], ticket["confidence"],
             ",".join(request.retrieved_ticket_ids)),
        )
        connection.commit()
    return {"ticket_id": ticket_id, "message": "Feedback captured for the next evaluation/training cycle."}
