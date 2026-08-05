// Serverless RAG demo client (T3.1).
//
// No build step and no framework (FR-3.2). The API is served from this same
// origin: CloudFront routes /v1/* to API Gateway and attaches the API key as an
// origin header, so there is no key here and no CORS preflight (ADR-008).

"use strict";

const API = "/v1";
const POLL_INTERVAL_MS = 3000;
const PENDING_STATUSES = new Set(["UPLOADED", "PROCESSING"]);

const el = {
  fileInput: document.getElementById("file-input"),
  uploadStatus: document.getElementById("upload-status"),
  documentsBody: document.getElementById("documents-body"),
  askForm: document.getElementById("ask-form"),
  askButton: document.getElementById("ask-button"),
  question: document.getElementById("question"),
  askStatus: document.getElementById("ask-status"),
  answer: document.getElementById("answer"),
  answerText: document.getElementById("answer-text"),
  answerMeta: document.getElementById("answer-meta"),
  citations: document.getElementById("citations"),
};

let pollTimer = null;

// --- API helpers -----------------------------------------------------------

async function api(method, path, body) {
  const response = await fetch(`${API}${path}`, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (response.status === 204) return null;

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error?.message || `Request failed (${response.status}).`);
  }
  return payload;
}

// --- Upload ----------------------------------------------------------------

async function upload(file) {
  setStatus(el.uploadStatus, `Preparing ${file.name}…`);

  const presigned = await api("POST", "/documents", { filename: file.name });

  if (file.size > presigned.max_bytes) {
    // The signed policy would reject this anyway; failing here is just faster
    // and clearer than a raw S3 error.
    await api("DELETE", `/documents/${presigned.doc_id}`).catch(() => {});
    throw new Error(
      `${file.name} is ${formatSize(file.size)}; the limit is ${formatSize(presigned.max_bytes)}.`
    );
  }

  const form = new FormData();
  // Order matters to S3: every signed field first, the bytes last. Content-Type
  // is deliberately not set on the request — the browser adds the multipart
  // boundary itself.
  for (const [name, value] of Object.entries(presigned.fields)) {
    form.append(name, value);
  }
  form.append("file", file);

  setStatus(el.uploadStatus, `Uploading ${file.name}…`);
  const uploaded = await fetch(presigned.upload_url, { method: "POST", body: form });

  if (!uploaded.ok) {
    // Leaving the metadata behind would show a document that will never index.
    await api("DELETE", `/documents/${presigned.doc_id}`).catch(() => {});
    throw new Error(`S3 rejected the upload (${uploaded.status}).`);
  }

  setStatus(el.uploadStatus, `${file.name} uploaded. Indexing…`);
  await refreshDocuments();
}

// --- Documents -------------------------------------------------------------

async function refreshDocuments() {
  let documents;
  try {
    ({ documents } = await api("GET", "/documents"));
  } catch (error) {
    renderMessage(`Could not load documents: ${error.message}`);
    return;
  }

  renderDocuments(documents);

  // Poll only while something is still being indexed, so an idle page is silent.
  clearTimeout(pollTimer);
  if (documents.some((document_) => PENDING_STATUSES.has(document_.status))) {
    pollTimer = setTimeout(refreshDocuments, POLL_INTERVAL_MS);
  }
}

function renderDocuments(documents) {
  el.documentsBody.replaceChildren();

  if (documents.length === 0) {
    renderMessage("No documents yet. Upload one to get started.");
    return;
  }

  for (const document_ of documents) {
    const row = document.createElement("tr");

    const counted = Boolean(document_.chunk_count);
    const chunks = cell(counted ? String(document_.chunk_count) : "—", "numeric");
    if (counted) {
      // Sits beside the number only while the column header is hidden, so a
      // narrow screen still says what the number counts.
      chunks.dataset.label = document_.chunk_count === 1 ? "chunk" : "chunks";
    }

    row.setAttribute("role", "row");
    row.append(
      cell(document_.filename, "name"),
      statusCell(document_),
      chunks,
      deleteCell(document_)
    );
    el.documentsBody.append(row);
  }
}

function cell(text, className) {
  const td = document.createElement("td");
  td.setAttribute("role", "cell");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function statusCell(document_) {
  const td = document.createElement("td");
  const badge = document.createElement("span");

  td.setAttribute("role", "cell");
  td.className = "state";

  badge.className = `badge ${document_.status.toLowerCase()}`;
  badge.textContent = document_.status;
  td.append(badge);

  if (document_.status === "FAILED" && document_.error_msg) {
    const reason = document.createElement("span");
    reason.className = "reason";
    reason.textContent = document_.error_msg;
    td.append(" ", reason);
  }
  return td;
}

function deleteCell(document_) {
  const td = document.createElement("td");
  const button = document.createElement("button");

  td.setAttribute("role", "cell");
  button.type = "button";
  button.className = "button subtle";
  button.textContent = "Delete";
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api("DELETE", `/documents/${document_.doc_id}`);
      await refreshDocuments();
    } catch (error) {
      setStatus(el.uploadStatus, error.message, true);
      button.disabled = false;
    }
  });

  td.append(button);
  td.className = "actions";
  return td;
}

function renderMessage(text) {
  const row = document.createElement("tr");
  const td = document.createElement("td");

  row.className = "empty";
  row.setAttribute("role", "row");
  td.setAttribute("role", "cell");
  td.colSpan = 4;
  td.textContent = text;
  row.append(td);
  el.documentsBody.replaceChildren(row);
}

// --- Ask -------------------------------------------------------------------

async function ask(question) {
  el.askButton.disabled = true;
  el.answer.hidden = true;
  setStatus(el.askStatus, "Searching your documents…");

  try {
    const result = await api("POST", "/query", { question });
    renderAnswer(result);
    setStatus(el.askStatus, "");
  } catch (error) {
    setStatus(el.askStatus, error.message, true);
  } finally {
    el.askButton.disabled = false;
  }
}

function renderAnswer(result) {
  el.answerText.textContent = result.answer;
  el.answerText.classList.toggle("not-found", !result.found);

  const parts = [`${result.latency_ms} ms`];
  if (result.found) {
    parts.push(`${result.usage.tokens_in} in / ${result.usage.tokens_out} out tokens`);
  } else {
    // Zero output tokens is the visible evidence that the similarity floor
    // short-circuited generation rather than the model declining.
    parts.push("no model call — below the similarity floor");
  }
  el.answerMeta.textContent = parts.join(" · ");

  el.citations.replaceChildren();
  for (const citation of result.citations) {
    el.citations.append(renderCitation(citation));
  }
  el.answer.hidden = false;
}

function renderCitation(citation) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  const score = document.createElement("span");
  const text = document.createElement("p");

  score.className = "score";
  score.textContent = citation.score.toFixed(3);
  summary.append(score, ` ${citation.filename} · ${citation.chunk_id}`);

  text.className = "chunk";
  text.textContent = citation.text;

  details.className = "citation";
  details.append(summary, text);
  return details;
}

// --- Wiring ----------------------------------------------------------------

function setStatus(node, message, isError = false) {
  node.textContent = message;
  node.classList.toggle("error", isError);
}

function formatSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

el.fileInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;

  try {
    await upload(file);
  } catch (error) {
    setStatus(el.uploadStatus, error.message, true);
  } finally {
    // Reset so choosing the same file again still fires a change event.
    event.target.value = "";
  }
});

el.askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = el.question.value.trim();
  if (question.length >= 3) ask(question);
});

refreshDocuments();
